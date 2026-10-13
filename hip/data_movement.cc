/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP data-movement acceptance tests: memcpy (each direction, sync + async),
// memset, the stream batch-memop path (including peer visibility of its
// writes), and the copy-engine batch copy.
// Built only when ACCEPT_WITH_HIP=ON; otherwise this file is empty.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// Source buffer holding buf[i] = i, used to seed copies below.
std::vector<int> make_indices(int n) {
  std::vector<int> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = i;
  return v;
}

// "" if the device buffer holds buf[i] == i for i in [0, n), else a reason
// naming `label` and the first offending element.
std::string indices_mismatch(const int* d_buf, int n, const char* label) {
  int bad_index = -1, bad_value = -1;
  if (hip_verify_indices(d_buf, n, &bad_index, &bad_value)) return "";
  return std::string(label) + " wrong at index " + std::to_string(bad_index) +
         ": got " + std::to_string(bad_value);
}

// Position-dependent byte pattern for the size sweep: a wrong size, a truncated
// tail, or a misaligned offset all surface as a byte mismatch at a known index.
// byte(i) = (i * stride + offset) mod 256; the odd, coprime-to-256 stride makes
// every position map to a distinct-looking value within any 256-byte window.
constexpr unsigned int kBytePatternStride = 131u;
constexpr unsigned int kBytePatternOffset = 7u;
void fill_byte_pattern(std::vector<unsigned char>& v) {
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<unsigned char>((i * kBytePatternStride +
                                       kBytePatternOffset) &
                                      0xFFu);
}

// Copy a device buffer back to host and compare against ref (the source device
// is inferred by the runtime from the pointer). Returns "" on match, else the
// first offending byte index.
std::string verify_device_bytes(const unsigned char* d,
                                 const std::vector<unsigned char>& ref) {
  std::vector<unsigned char> back(ref.size(), 0);
  if (hipMemcpy(back.data(), d, ref.size(), hipMemcpyDeviceToHost) != hipSuccess)
    return "D2H verify copy failed";
  for (size_t i = 0; i < ref.size(); ++i)
    if (back[i] != ref[i])
      return "mismatch at byte " + std::to_string(i) + " (got " +
             std::to_string(back[i]) + ", want " + std::to_string(ref[i]) + ")";
  return "";
}

std::string sweep_one_size(size_t bytes, bool async, int peer, PerfSink& perf) {
  std::vector<unsigned char> reference(bytes);
  fill_byte_pattern(reference);

  const std::string mode = async ? "async" : "sync";
  const std::string size_tag = " (S=" + std::to_string(bytes) + ", " + mode + ")";

  if (hipSetDevice(0) != hipSuccess)
    return "hipSetDevice(0)" + size_tag;

  Stream stream;
  if (async && stream.create() != hipSuccess)
    return "hipStreamCreate" + size_tag;
  hipStream_t active_stream = async ? stream.get() : nullptr;

  DeviceBuffer source;        // GPU 0 source
  DeviceBuffer dest_d2d;      // GPU 0 D2D destination
  DeviceBuffer dest_peer;     // peer GPU P2P destination
  if (source.allocate(bytes) != hipSuccess)
    return "hipMalloc(source)" + size_tag;
  if (dest_d2d.allocate(bytes) != hipSuccess)
    return "hipMalloc(dest_d2d)" + size_tag;

  // H2D, then verify via D2H (covers both host<->device directions).
  hipError_t copy_status =
      perf_memcpy(perf, "h2d." + mode, source.get(), reference.data(), bytes,
                  hipMemcpyHostToDevice, active_stream);
  if (copy_status != hipSuccess)
    return "H2D: " + std::string(hipGetErrorString(copy_status)) + size_tag;
  std::string verify_error =
      verify_device_bytes(source.as<unsigned char>(), reference);
  if (!verify_error.empty())
    return "H2D/D2H " + verify_error + size_tag;

  // D2D on GPU 0.
  copy_status = perf_memcpy(perf, "d2d." + mode, dest_d2d.get(), source.get(),
                            bytes, hipMemcpyDeviceToDevice, active_stream);
  if (copy_status != hipSuccess)
    return "D2D: " + std::string(hipGetErrorString(copy_status)) + size_tag;
  verify_error = verify_device_bytes(dest_d2d.as<unsigned char>(), reference);
  if (!verify_error.empty())
    return "D2D " + verify_error + size_tag;

  // P2P (GPU 0 -> peer) when a peer is available.
  if (peer >= 0) {
    if (hipSetDevice(peer) != hipSuccess)
      return "hipSetDevice(peer)" + size_tag;
    if (dest_peer.allocate(bytes) != hipSuccess)
      return "hipMalloc(peer)" + size_tag;
    (void)hipSetDevice(0);

    copy_status = perf_memcpy(perf, "p2p." + mode, dest_peer.get(),
                              source.get(), bytes, hipMemcpyDeviceToDevice,
                              active_stream, peer, /*src_device=*/0);
    if (copy_status != hipSuccess)
      return "P2P: " + std::string(hipGetErrorString(copy_status)) + size_tag;
    verify_error = verify_device_bytes(dest_peer.as<unsigned char>(), reference);
    if (!verify_error.empty())
      return "P2P " + verify_error + size_tag;
  }
  return "";
}

// Direction for the six single-size index-copy tests below.
enum class CopyDir { H2D, D2D, D2H };

// Shared body for the six direction x {sync,async} index-copy tests: stage
// make_indices(n) into the source, copy it to the destination in `dir` (async
// on a fresh stream when `async`), then verify the destination holds 0..n-1.
// `label` is folded into the failure message. Sync uses the blocking
// hipMemcpy; async uses hipMemcpyAsync + hipStreamSynchronize.
TestResult run_index_memcpy_test(CopyDir dir, bool async, const char* label) {
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  const std::vector<int> host_src = make_indices(n);

  Stream stream;
  if (async) HIP_CHECK(stream.create());
  auto do_copy = [&](void* dst, const void* src,
                     hipMemcpyKind kind) -> std::string {
    if (async) {
      hipError_t status = hipMemcpyAsync(dst, src, bytes, kind, stream.get());
      if (status != hipSuccess)
        return std::string("hipMemcpyAsync: ") + hipGetErrorString(status);
      status = hipStreamSynchronize(stream.get());
      if (status != hipSuccess)
        return std::string("hipStreamSynchronize: ") + hipGetErrorString(status);
      return "";
    }
    hipError_t status = hipMemcpy(dst, src, bytes, kind);
    if (status != hipSuccess)
      return std::string("hipMemcpy: ") + hipGetErrorString(status);
    return "";
  };

  if (dir == CopyDir::D2H) {
    DeviceBuffer src;
    HIP_CHECK(src.allocate(bytes));
    HIP_CHECK(
        hipMemcpy(src.get(), host_src.data(), bytes, hipMemcpyHostToDevice));
    std::vector<int> host_dst(static_cast<size_t>(n), -1);
    if (const std::string e =
            do_copy(host_dst.data(), src.get(), hipMemcpyDeviceToHost);
        !e.empty())
      return fail(e);
    const std::string err = verify_host_indices(host_dst.data(), n);
    if (!err.empty()) return fail(std::string(label) + " " + err);
    return pass();
  }

  DeviceBuffer dst;
  HIP_CHECK(dst.allocate(bytes));
  HIP_CHECK(hipMemset(dst.get(), kPoisonByte, bytes));  // catch a no-op copy
  if (dir == CopyDir::H2D) {
    if (const std::string e =
            do_copy(dst.get(), host_src.data(), hipMemcpyHostToDevice);
        !e.empty())
      return fail(e);
  } else {  // CopyDir::D2D — stage a device source first, then device->device.
    DeviceBuffer src;
    HIP_CHECK(src.allocate(bytes));
    HIP_CHECK(
        hipMemcpy(src.get(), host_src.data(), bytes, hipMemcpyHostToDevice));
    if (const std::string e =
            do_copy(dst.get(), src.get(), hipMemcpyDeviceToDevice);
        !e.empty())
      return fail(e);
  }
  const std::string err = indices_mismatch(dst.as<int>(), n, label);
  if (!err.empty()) return fail(err);
  return pass();
}

// How the producer signals that the payload is written.
enum class FlagWriter { BatchMemOp, WriteValue32 };

// Iterations per producer->consumer pair. Where the bug reproduces the flag
// overtakes the payload in 20-100% of iterations, so 200 cannot miss it.
constexpr int kPeerSignalIters = 200;
constexpr int kPeerPayloadBase = 1000;

hipError_t write_flag(hipStream_t stream, uint32_t* flag, uint32_t value,
                      FlagWriter writer) {
  if (writer == FlagWriter::WriteValue32)
    return hipStreamWriteValue32(stream, flag, value, 0);
  hipStreamBatchMemOpParams op{};
  op.writeValue.operation = hipStreamMemOpWriteValue32;
  op.writeValue.address = reinterpret_cast<hipDeviceptr_t>(flag);
  op.writeValue.value = value;
  return hipStreamBatchMemOp(stream, 1, &op, 0);
}

struct StaleReads {
  int count = 0;
  int from_previous_iteration = 0;
};

// Each iteration the producer writes a fresh payload into consumer memory and
// then the flag, on one stream; the consumer waits for the flag and copies the
// payload out. Counts the iterations that copied anything else into `stale`.
TestResult count_stale_reads(int producer, int consumer, FlagWriter writer,
                             StaleReads& stale) {
  HIP_CHECK(hipSetDevice(consumer));
  DeviceBuffer payload_buf, flag, result;
  HIP_CHECK(payload_buf.allocate(sizeof(int)));
  HIP_CHECK(flag.allocate(sizeof(uint32_t)));
  HIP_CHECK(result.allocate(sizeof(int)));
  HIP_CHECK(hipMemset(flag.get(), 0, sizeof(uint32_t)));
  HIP_CHECK(hipDeviceSynchronize());
  Stream consumer_stream, producer_stream;
  HIP_CHECK(consumer_stream.create_with_flags(hipStreamNonBlocking));
  HIP_CHECK(hipSetDevice(producer));
  HIP_CHECK(producer_stream.create_with_flags(hipStreamNonBlocking));

  for (int i = 0; i < kPeerSignalIters; ++i) {
    const uint32_t flag_value = static_cast<uint32_t>(i + 1);
    const int payload = kPeerPayloadBase + i;

    HIP_CHECK(hipSetDevice(consumer));
    HIP_CHECK(hipStreamWaitValue32(consumer_stream.get(), flag.get(),
                                   flag_value, hipStreamWaitValueEq,
                                   0xFFFFFFFFu));
    HIP_CHECK(hipMemcpyAsync(result.get(), payload_buf.get(), sizeof(int),
                             hipMemcpyDeviceToDevice, consumer_stream.get()));

    HIP_CHECK(hipSetDevice(producer));
    HIP_CHECK(
        hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(payload_buf.get()),
                          payload, 1, producer_stream.get()));
    HIP_CHECK(write_flag(producer_stream.get(), flag.as<uint32_t>(), flag_value,
                         writer));
    HIP_CHECK(hipStreamSynchronize(producer_stream.get()));

    HIP_CHECK(hipSetDevice(consumer));
    HIP_CHECK(hipStreamSynchronize(consumer_stream.get()));
    int got = 0;
    HIP_CHECK(
        hipMemcpy(&got, result.get(), sizeof(int), hipMemcpyDeviceToHost));
    if (got == payload) continue;
    ++stale.count;
    if (got == payload - 1) ++stale.from_previous_iteration;
  }
  return pass();
}

// Shared body for the two *_PeerVisibility tests: runs count_stale_reads over
// every peer-capable ordered GPU pair and fails naming each pair that read
// stale data.
TestResult run_peer_visibility_test(FlagWriter writer) {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 2)
    return skip("only " + std::to_string(count) + " GPU(s); no peer pairs");

  std::string failures;
  int pairs = 0;
  for (int producer = 0; producer < count; ++producer) {
    for (int consumer = 0; consumer < count; ++consumer) {
      if (producer == consumer) continue;
      int peer_accessible = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&peer_accessible, producer, consumer));
      if (!peer_accessible) continue;
      const std::string peer_err =
          enable_peer_access_idempotent(producer, consumer);
      if (!peer_err.empty()) return fail(peer_err);

      const std::string pair =
          std::to_string(producer) + "->" + std::to_string(consumer);
      StaleReads stale;
      TestResult pair_result =
          count_stale_reads(producer, consumer, writer, stale);
      if (pair_result.status == Status::Fail) {
        pair_result.reason = pair + ": " + pair_result.reason;
        return pair_result;
      }
      ++pairs;
      if (stale.count == 0) continue;
      if (!failures.empty()) failures += "; ";
      failures += pair + ": " + std::to_string(stale.count) + "/" +
                  std::to_string(kPeerSignalIters) + " stale reads (" +
                  std::to_string(stale.from_previous_iteration) +
                  " from the previous iteration)";
    }
  }
  if (pairs == 0) return skip("no GPU pair reports peer access");
  if (!failures.empty()) return fail(failures);
  return pass();
}

}  // namespace

// Sync host-to-device copy via hipMemcpy; verify the device buffer holds the source indices.
ACCEPT_TEST(Memcpy_H2D_Sync, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return run_index_memcpy_test(CopyDir::H2D, /*async=*/false, "H2D");
}

// Async host-to-device copy on a stream; sync and verify the device buffer.
ACCEPT_TEST(Memcpy_H2D_Async, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return run_index_memcpy_test(CopyDir::H2D, /*async=*/true, "H2D async");
}

// Sync device-to-device copy via hipMemcpy; verify the destination buffer.
ACCEPT_TEST(Memcpy_D2D_Sync, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return run_index_memcpy_test(CopyDir::D2D, /*async=*/false, "D2D");
}

// Async device-to-device copy on a stream; sync and verify the destination buffer.
ACCEPT_TEST(Memcpy_D2D_Async, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return run_index_memcpy_test(CopyDir::D2D, /*async=*/true, "D2D async");
}

// Sync device-to-host copy via hipMemcpy; verify indices on the host.
ACCEPT_TEST(Memcpy_D2H_Sync, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return run_index_memcpy_test(CopyDir::D2H, /*async=*/false, "D2H");
}

// Async device-to-host copy on a stream; sync and verify indices on the host.
ACCEPT_TEST(Memcpy_D2H_Async, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return run_index_memcpy_test(CopyDir::D2H, /*async=*/true, "D2H async");
}

// Sweep increasing copy sizes (1 B .. 256 MiB, including an odd-byte size)
// across every direction -- H2D, D2H, D2D (same GPU) and P2P (cross-GPU) -- in
// both sync and async modes, verifying a position-dependent byte pattern each
// time. Catches size-threshold / alignment / staging-path bugs that a single
// fixed size misses. The P2P leg is skipped when there is no peer-capable
// second GPU; the rest still runs on a single-GPU node.
ACCEPT_TEST(Memcpy_Size_Sweep, "hip/datamovement", Topology::Single,
            Kind::Functional, 180) {
  (void)ctx;
  HIP_CHECK(hipSetDevice(0));

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  int peer = -1;
  for (int device = 1; device < device_count && peer < 0; ++device) {
    int can_access = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access, 0, device));
    if (can_access)
      peer = device;
  }
  if (peer >= 0) {
    const std::string peer_err = enable_peer_access_idempotent(0, peer);
    if (!peer_err.empty()) return fail(peer_err);
  }

  const size_t sizes[] = {1u,
                          3u,
                          1024u,
                          64u * 1024u,
                          1024u * 1024u,
                          16u * 1024u * 1024u,
                          256u * 1024u * 1024u};
  // Functional; under perf mode each leg's bandwidth rides out on result.perf.
  // Each size is swept in both the sync and the async transfer mode.
  TestResult result = pass();
  for (size_t bytes : sizes) {
    for (bool async : {false, true}) {
      const std::string err = sweep_one_size(bytes, async, peer, result.perf);
      if (!err.empty())
        return fail(err);
    }
  }
  return result;
}

// Zero a poisoned device buffer with hipMemset and confirm every byte reads back 0.
ACCEPT_TEST(Memset_Sync, "hip/datamovement", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  const size_t bytes = 4096;

  DeviceBuffer buf;
  HIP_CHECK(buf.allocate(bytes));
  HIP_CHECK(hipMemset(buf.get(), kPoisonByte, bytes));  // poison so a miss shows
  HIP_CHECK(hipMemset(buf.get(), 0, bytes));

  std::vector<uint8_t> host(bytes, kPoisonByte);
  HIP_CHECK(hipMemcpy(host.data(), buf.get(), bytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < bytes; ++i)
    if (host[i] != 0)
      return fail("hipMemset left nonzero byte at " + std::to_string(i) + ": " +
                  std::to_string(host[i]));
  return pass();
}

// Zero a poisoned device buffer with hipMemsetAsync on a stream and confirm every byte is 0.
ACCEPT_TEST(Memset_Async, "hip/datamovement", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  const size_t bytes = 4096;

  DeviceBuffer buf;
  Stream stream;
  HIP_CHECK(buf.allocate(bytes));
  HIP_CHECK(stream.create());
  HIP_CHECK(hipMemset(buf.get(), kPoisonByte, bytes));
  HIP_CHECK(hipMemsetAsync(buf.get(), 0, bytes, stream.get()));
  HIP_CHECK(hipStreamSynchronize(stream.get()));

  std::vector<uint8_t> host(bytes, kPoisonByte);
  HIP_CHECK(hipMemcpy(host.data(), buf.get(), bytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < bytes; ++i)
    if (host[i] != 0)
      return fail("hipMemsetAsync left nonzero byte at " + std::to_string(i) +
                  ": " + std::to_string(host[i]));
  return pass();
}

// Enqueue batch writeValue32 ops on a stream and confirm each device word becomes 0.
ACCEPT_TEST(StreamBatchMemOp_WriteValue, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  // If the running ROCm lacks hipStreamBatchMemOp the call below trips the
  // checked_op guard (MissingSymbol -> Skip "requires hipStreamBatchMemOp").
  const unsigned int word_count = 8;
  const size_t bytes = sizeof(uint32_t) * word_count;

  DeviceBuffer words;
  Stream stream;
  HIP_CHECK(words.allocate(bytes));
  HIP_CHECK(stream.create());
  HIP_CHECK(hipMemset(words.get(), kPoisonByte, bytes));  // writing 0 is visible
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<hipStreamBatchMemOpParams> params(word_count);
  for (unsigned int i = 0; i < word_count; ++i) {
    params[i].writeValue.operation = hipStreamMemOpWriteValue32;
    params[i].writeValue.address =
        reinterpret_cast<hipDeviceptr_t>(words.as<uint32_t>() + i);
    params[i].writeValue.value = 0;
    params[i].writeValue.flags = 0;
  }

  HIP_CHECK(hipStreamBatchMemOp(stream.get(), word_count, params.data(), 0));
  HIP_CHECK(hipStreamSynchronize(stream.get()));

  std::vector<uint32_t> host(word_count, 0xFFFFFFFFu);
  HIP_CHECK(hipMemcpy(host.data(), words.get(), bytes, hipMemcpyDeviceToHost));

  for (unsigned int i = 0; i < word_count; ++i)
    if (host[i] != 0)
      return fail("batch writeValue word " + std::to_string(i) +
                  " not written: got " + std::to_string(host[i]));
  return pass();
}

// A flag written by hipStreamBatchMemOp must not become visible to a peer
// device before data written earlier on the same stream (ROCM-31976), checked
// over every peer-capable ordered GPU pair. Skips on single-GPU nodes.
ACCEPT_TEST(StreamBatchMemOp_WriteValue_PeerVisibility, "hip/datamovement",
            Topology::Single, Kind::Functional, 120) {
  (void)ctx;
  return run_peer_visibility_test(FlagWriter::BatchMemOp);
}

// Control for StreamBatchMemOp_WriteValue_PeerVisibility: the same handoff
// with the flag written by hipStreamWriteValue32.
ACCEPT_TEST(StreamWriteValue_PeerVisibility, "hip/datamovement",
            Topology::Single, Kind::Functional, 120) {
  (void)ctx;
  return run_peer_visibility_test(FlagWriter::WriteValue32);
}

// Run hipMemcpyBatchAsync then a dependent D2D copy ordered by an event; verify integrity.
ACCEPT_TEST(Ce_BatchCopy_Ordered, "hip/datamovement", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  // If the running ROCm lacks hipMemcpyBatchAsync the call below trips the
  // checked_op guard (MissingSymbol -> Skip "requires hipMemcpyBatchAsync").
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  const std::vector<int> host_src = make_indices(n);

  DeviceBuffer src, dst;
  Stream stream;
  Event ordering_event;
  HIP_CHECK(src.allocate(bytes));
  HIP_CHECK(dst.allocate(bytes));
  HIP_CHECK(stream.create());
  HIP_CHECK(ordering_event.create());

  void* dsts[1] = {src.get()};
  void* srcs[1] = {const_cast<int*>(host_src.data())};
  size_t sizes[1] = {bytes};

  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagPreferOverlapWithCompute;

  size_t attr_idxs[1] = {0};
  size_t fail_idx = 0;
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, /*count=*/1, &attr, attr_idxs,
                                /*numAttrs=*/1, &fail_idx, stream.get()));

  // Chain a dependent D2D copy after the batch via the event.
  HIP_CHECK(hipEventRecord(ordering_event.get(), stream.get()));
  HIP_CHECK(hipStreamWaitEvent(stream.get(), ordering_event.get(), 0));
  HIP_CHECK(hipMemcpyAsync(dst.get(), src.get(), bytes, hipMemcpyDeviceToDevice,
                           stream.get()));
  HIP_CHECK(hipStreamSynchronize(stream.get()));

  const std::string err = indices_mismatch(dst.as<int>(), n, "CE batch-copy chain");
  if (!err.empty()) return fail(err);
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
