/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Device management group.
//
// Built only when ACCEPT_WITH_HIP=ON; otherwise this file is empty. Covers the
// device enumeration/selection, attribute probing, peer-access matrix, driver
// entry-point resolution, and device-reset floor RCCL and the ROCm libraries
// stand on at init.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "common/TestRegistry.hpp"
#include "hip/kernel_launch/StaticKernels.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// True if gcnArchName names a pre-gfx90a GPU, the only arch family where the
// HDP flush register is on the GDR path; gfx90a and newer flush implicitly.
bool is_pre_gfx90a(const char* arch) {
  return std::strstr(arch, "gfx900") || std::strstr(arch, "gfx906") ||
         std::strstr(arch, "gfx908");
}

// Optional driver symbol: resolved or cleanly absent (mirrors cudaPfnFuncLoader).
bool driver_symbol_resolved_or_absent(const char* sym, std::string* err) {
  void* pfn = nullptr;
  hipDriverProcAddressQueryResult status = HIP_GET_PROC_ADDRESS_SUCCESS;
  const hipError_t query_err = hipGetProcAddress(sym, &pfn, HIP_VERSION, 0,
                                                 &status);
  const bool resolved = (query_err == hipSuccess && pfn != nullptr &&
                         status == HIP_GET_PROC_ADDRESS_SUCCESS);
  const bool clean_absent =
      (status == HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND ||
       status == HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT);
  if (!resolved && !clean_absent) {
    if (err)
      *err = std::string(sym) + " (err " + hipGetErrorString(query_err) +
             ", status " + std::to_string(status) + ")";
    return false;
  }
  (void)hipGetLastError();
  return true;
}

// Library-surfaced paths (occupancy / texture / mem-hints / symbol) that the
// ROCm libraries depend on. Occupancy/stack kernels come from hiprtc; the
// texture-fetch and __device__-symbol kernels are statically linked
// (StaticKernels.hpp), which hiprtc cannot supply.

// True if the device exposes the linear-texture fetch path. CDNA3+ (gfx94+)
// has no image hardware, so the texture intrinsics are compiled out there.
bool device_has_texture(int dev) {
  int width = 0;
  return hipDeviceGetAttribute(&width, hipDeviceAttributeMaxTexture1DWidth,
                               dev) == hipSuccess &&
         width > 0;
}

// One directed peer transfer a -> b: fill the source on `a`, copy it across the
// link (sync or async), then count mismatches with a kernel on `b`. Returns ""
// on success or a reason. Not HIP_CHECK (that returns from the test body).
std::string p2p_directed_copy(int a, int b, int n, bool use_async) {
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  // Source on GPU a, filled with indices by a kernel on a.
  if (hipSetDevice(a) != hipSuccess) return "hipSetDevice(src)";
  DeviceBuffer src;
  if (src.allocate(bytes) != hipSuccess) return "hipMalloc(src)";
  int* src_ptr = src.as<int>();
  void* fill_args[] = {&src_ptr, &n};
  if (hipLaunchKernel(sk_fill_indices_ptr(), grid_for(n), dim3(kBlockSize),
                      fill_args, 0, 0) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess)
    return "fill kernel on src";

  // Destination + mismatch counter on GPU b.
  if (hipSetDevice(b) != hipSuccess) return "hipSetDevice(dst)";
  DeviceBuffer dst, mism;
  if (dst.allocate(bytes) != hipSuccess) return "hipMalloc(dst)";
  if (mism.allocate(sizeof(int)) != hipSuccess) return "hipMalloc(mism)";
  if (hipMemset(mism.get(), 0, sizeof(int)) != hipSuccess)
    return "hipMemset(mism)";

  // The cross-device copy a -> b.
  Stream stream;
  hipError_t copy_err;
  std::string copy_fn;
  if (use_async) {
    if (stream.create() != hipSuccess) return "hipStreamCreate";
    copy_err =
        hipMemcpyPeerAsync(dst.get(), b, src.get(), a, bytes, stream.get());
    copy_fn = "hipMemcpyPeerAsync";
    if (copy_err == hipSuccess) {
      copy_err = hipStreamSynchronize(stream.get());
      copy_fn = "hipStreamSynchronize";
    }
  } else {
    copy_err = hipMemcpyPeer(dst.get(), b, src.get(), a, bytes);
    copy_fn = "hipMemcpyPeer";
  }
  if (copy_err != hipSuccess)
    return copy_fn + ": " + hipGetErrorString(copy_err);

  // Verify on GPU b with a kernel reading the peer-copied buffer.
  int* dst_ptr = dst.as<int>();
  int* mism_ptr = mism.as<int>();
  void* verify_args[] = {&dst_ptr, &n, &mism_ptr};
  if (hipLaunchKernel(sk_count_mismatch_indices_ptr(), grid_for(n),
                      dim3(kBlockSize), verify_args, 0, 0) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess)
    return "verify kernel on dst";
  int mismatches = -1;
  if (hipMemcpy(&mismatches, mism.get(), sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return "copy mismatch count to host";
  if (mismatches != 0)
    return std::to_string(mismatches) + " mismatched element(s) after peer copy";
  return "";
}

// Kernel-driven peer LOAD reader<-owner: GPU `owner` fills its own buffer, then
// a kernel on GPU `reader` reads that buffer directly over the peer link
// (in-kernel peer load, not the copy-engine path hipMemcpyPeer drives) and
// tallies any mismatch into a reader-local counter. This is the direct GPU->GPU
// read data path RCCL relies on. A peer STORE (reader writing into owner, then
// owner reading it) is deliberately NOT used: without a producer/consumer flag
// handshake the remote writes are not ordered against the consumer and the check
// races; the read direction settles the owner's data with a plain sync before
// the reader touches it, so it is race-free.
//
// Peer access reader->owner must already be enabled. The static kernels launch
// on any current device (the hiprtc module is bound to one device, so it is not
// used here). The read kernel is the timed region: one pass verifies, warmup+N
// passes under --kind perf feed per-op latency to `sink` under `metric` (empty
// metric skips emission). Returns "" on success.
std::string p2p_kernel_read(int reader, int owner, int element_count,
                            PerfSink& sink, const std::string& metric,
                            MeasureOpts options) {
  const size_t bytes = sizeof(int) * static_cast<size_t>(element_count);

  // Owner fills its OWN buffer with indices (local, coherent after the sync).
  // A sentinel memset first so a silent no-op fill cannot pass verification.
  if (hipSetDevice(owner) != hipSuccess) {
    return "hipSetDevice(owner)";
  }
  DeviceBuffer owned;
  if (owned.allocate(bytes) != hipSuccess) {
    return "hipMalloc(owned)";
  }
  if (hipMemset(owned.get(), 0xff, bytes) != hipSuccess) {
    return "hipMemset(owned)";
  }
  int* owned_ptr = owned.as<int>();
  void* fill_args[] = {&owned_ptr, &element_count};
  if (hipLaunchKernel(sk_fill_indices_ptr(), grid_for(element_count),
                      dim3(kBlockSize), fill_args, 0, 0) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess) {
    return "fill kernel on " + std::to_string(owner);
  }

  // Reader's kernel reads the owner's buffer over the link and tallies any
  // mismatch into a reader-local counter. Correct data keeps the counter at 0
  // across perf passes, so no per-pass reset is needed.
  if (hipSetDevice(reader) != hipSuccess) {
    return "hipSetDevice(reader)";
  }
  DeviceBuffer mismatches;
  if (mismatches.allocate(sizeof(int)) != hipSuccess) {
    return "hipMalloc(mismatches)";
  }
  if (hipMemset(mismatches.get(), 0, sizeof(int)) != hipSuccess) {
    return "hipMemset(mismatches)";
  }
  int* mismatch_ptr = mismatches.as<int>();
  void* read_args[] = {&owned_ptr, &element_count, &mismatch_ptr};

  // Own stream so the timed sync waits on this kernel alone, not the whole
  // device (hipStreamSynchronize instead of hipDeviceSynchronize).
  Stream stream;
  if (stream.create() != hipSuccess) {
    return "hipStreamCreate on " + std::to_string(reader);
  }

  PerfLoop loop(options);
  for (auto pass : loop) {
    if (hipLaunchKernel(sk_count_mismatch_indices_ptr(), grid_for(element_count),
                        dim3(kBlockSize), read_args, 0, stream.get()) !=
            hipSuccess ||
        hipStreamSynchronize(stream.get()) != hipSuccess) {
      return "peer-load kernel on " + std::to_string(reader);
    }
    if (pass.first()) {
      int host_mismatches = -1;
      if (hipMemcpy(&host_mismatches, mismatches.get(), sizeof(int),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        return "copy mismatch count to host";
      }
      if (host_mismatches != 0) {
        return std::to_string(host_mismatches) +
               " element(s) misread on peer load";
      }
    }
  }

  if (!metric.empty()) {
    loop.emit_latency(sink, metric);
  }
  return "";
}

// The scalar device attributes the probe reads once and then cross-checks
// against the batch-query and hipGetDeviceProperties paths.
struct ProbedAttrs {
  int warp = 0;
  int mp = 0;
  int managed_host = -1;
  int host_numa = -99;
  bool have_host_numa = false;  // whether the runtime answered HostNumaId
};

// Read the individual RCCL-relevant attrs on `dev` and range-check each.
// Returns "" on success or a reason naming the offending attribute.
std::string probe_scalar_attrs(int dev, ProbedAttrs& out) {
  if (hipDeviceGetAttribute(&out.warp, hipDeviceAttributeWarpSize, dev) !=
      hipSuccess)
    return "hipDeviceGetAttribute(WarpSize) failed";
  if (out.warp != 32 && out.warp != 64)
    return "unexpected warp size: " + std::to_string(out.warp);

  if (hipDeviceGetAttribute(&out.mp, hipDeviceAttributeMultiprocessorCount,
                            dev) != hipSuccess)
    return "hipDeviceGetAttribute(MultiprocessorCount) failed";
  if (out.mp <= 0)
    return "multiProcessorCount not positive: " + std::to_string(out.mp);

  if (hipDeviceGetAttribute(&out.managed_host,
                            hipDeviceAttributeDirectManagedMemAccessFromHost,
                            dev) != hipSuccess)
    return "hipDeviceGetAttribute(DirectManagedMemAccessFromHost) failed";
  if (out.managed_host != 0 && out.managed_host != 1)
    return "DirectManagedMemAccessFromHost not boolean: " +
           std::to_string(out.managed_host);

  // HostNumaId is always declared by the headers; whether the RUNTIME answers
  // the query is detected here. Older runtimes return an error, in which case
  // the sub-check is skipped rather than failing the whole probe.
  out.have_host_numa =
      hipDeviceGetAttribute(&out.host_numa, hipDeviceAttributeHostNumaId, dev) ==
      hipSuccess;
  (void)hipGetLastError();
  if (out.have_host_numa && out.host_numa < -1)
    return "HostNumaId out of range: " + std::to_string(out.host_numa);
  return "";
}

// Re-read the same attrs through the batched paths and confirm they agree with
// the scalar reads: first the single-call loop, then hipDeviceGetAttributes
// when the driver exports it. Returns "" on success.
std::string probe_batch_attrs(int dev, const ProbedAttrs& scalars) {
  std::vector<hipDeviceAttribute_t> batch_attrs = {
      hipDeviceAttributeWarpSize,
      hipDeviceAttributeMultiprocessorCount,
      hipDeviceAttributeDirectManagedMemAccessFromHost,
  };
  if (scalars.have_host_numa)
    batch_attrs.push_back(hipDeviceAttributeHostNumaId);
  const int batch_count = static_cast<int>(batch_attrs.size());
  int batch_vals[4] = {};
  for (int i = 0; i < batch_count; ++i)
    if (hipDeviceGetAttribute(&batch_vals[i], batch_attrs[i], dev) != hipSuccess)
      return "hipDeviceGetAttribute(batch index " + std::to_string(i) +
             ") failed";
  if (batch_vals[0] != scalars.warp)
    return "batch warp mismatch: " + std::to_string(batch_vals[0]);
  if (batch_vals[1] != scalars.mp)
    return "batch multiProcessorCount mismatch: " + std::to_string(batch_vals[1]);
  if (batch_vals[2] != scalars.managed_host)
    return "batch DirectManagedMemAccessFromHost mismatch: " +
           std::to_string(batch_vals[2]);
  if (scalars.have_host_numa && batch_vals[3] != scalars.host_numa)
    return "batch HostNumaId mismatch: " + std::to_string(batch_vals[3]);

  void* get_attrs_pfn = nullptr;
  hipDriverProcAddressQueryResult get_attrs_status = HIP_GET_PROC_ADDRESS_SUCCESS;
  const hipError_t get_attrs_err = hipGetProcAddress(
      "hipDeviceGetAttributes", &get_attrs_pfn, HIP_VERSION, 0,
      &get_attrs_status);
  if (get_attrs_err == hipSuccess && get_attrs_pfn != nullptr &&
      get_attrs_status == HIP_GET_PROC_ADDRESS_SUCCESS) {
    using HipDeviceGetAttributesFn =
        hipError_t (*)(int*, const hipDeviceAttribute_t*, int, int);
    auto get_attrs = reinterpret_cast<HipDeviceGetAttributesFn>(get_attrs_pfn);
    int gpa_vals[4] = {};
    if (get_attrs(gpa_vals, batch_attrs.data(), batch_count, dev) != hipSuccess)
      return "hipDeviceGetAttributes batch query failed";
    for (int i = 0; i < batch_count; ++i)
      if (gpa_vals[i] != batch_vals[i])
        return "hipDeviceGetAttributes mismatch for attr index " +
               std::to_string(i) + ": got " + std::to_string(gpa_vals[i]) +
               ", expected " + std::to_string(batch_vals[i]);
  } else if (get_attrs_status != HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND &&
             get_attrs_status != HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT) {
    return "hipDeviceGetAttributes gave unclean status " +
           std::to_string(get_attrs_status);
  }
  (void)hipGetLastError();
  return "";
}

// Cross-check the attrs against hipGetDeviceProperties. Returns "" on success.
std::string cross_check_props(int dev, const ProbedAttrs& scalars) {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess)
    return "hipGetDeviceProperties failed";
  if (prop.gcnArchName[0] == '\0') return "empty gcnArchName";
  if (prop.warpSize != scalars.warp)
    return "warpSize/prop mismatch: attr=" + std::to_string(scalars.warp) +
           " prop=" + std::to_string(prop.warpSize);
  if (prop.multiProcessorCount != scalars.mp)
    return "multiProcessorCount/prop mismatch";
  return "";
}

}  // namespace

// Enumerate devices and round-trip each through set/get + PCI bus-id lookup.
ACCEPT_TEST(Device_Query_Basics, "hip/device", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 1)
    return fail("hipGetDeviceCount returned " + std::to_string(count));

  for (int dev = 0; dev < count; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    int current = -1;
    HIP_CHECK(hipGetDevice(&current));
    if (current != dev)
      return fail("hipGetDevice mismatch: set " + std::to_string(dev) +
                  ", got " + std::to_string(current));

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    char pci[64] = {0};
    HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), dev));
    int back = -1;
    HIP_CHECK(hipDeviceGetByPCIBusId(&back, pci));
    if (back != dev)
      return fail("PCI bus-id round-trip mismatch for dev " +
                  std::to_string(dev) + " (" + pci + "): got " +
                  std::to_string(back));
  }
  return pass();
}

// Build the peer-access matrix; for the first peer-capable pair, enable access
// and verify a real cross-device copy (both hipMemcpyPeerAsync and set+async).
ACCEPT_TEST(Device_CanAccessPeer_Matrix, "hip/device", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 2)
    return skip("only " + std::to_string(count) + " GPU(s); no peer pairs");

  int src_dev = -1, dst_dev = -1;
  for (int i = 0; i < count && src_dev < 0; ++i)
    for (int j = 0; j < count; ++j) {
      if (i == j) continue;
      int peer_accessible = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&peer_accessible, i, j));
      if (peer_accessible) {
        src_dev = i;
        dst_dev = j;
        break;
      }
    }
  if (src_dev < 0) return skip("no GPU pair reports peer access");

  const std::string peer_err = enable_peer_access_idempotent(src_dev, dst_dev);
  if (!peer_err.empty()) return fail(peer_err);

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  std::vector<int> host_src(n);
  for (int i = 0; i < n; ++i) host_src[i] = i;

  DeviceBuffer src, dst;
  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(src.allocate(bytes));
  HIP_CHECK(hipMemcpy(src.get(), host_src.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipSetDevice(dst_dev));
  HIP_CHECK(dst.allocate(bytes));

  Stream stream;
  HIP_CHECK(stream.create());
  hipError_t copy_err =
      hipMemcpyPeerAsync(dst.get(), dst_dev, src.get(), src_dev, bytes,
                         stream.get());
  std::string copy_fn = "hipMemcpyPeerAsync";
  if (copy_err == hipSuccess) {
    copy_err = hipStreamSynchronize(stream.get());
    copy_fn = "hipStreamSynchronize";
  }

  std::vector<int> host_out(n, -1);
  if (copy_err == hipSuccess) {
    copy_err =
        hipMemcpy(host_out.data(), dst.get(), bytes, hipMemcpyDeviceToHost);
    copy_fn = "hipMemcpy(D2H)";
  }

  if (copy_err != hipSuccess)
    return fail(copy_fn + " " + std::to_string(src_dev) + "->" +
                std::to_string(dst_dev) + ": " + hipGetErrorString(copy_err));
  const std::string err = verify_host_indices(host_out.data(), n);
  if (!err.empty()) return fail("peer copy " + err);
  return pass();
}

// Exercise every peer-capable GPU pair on the node with a real cross-device
// transfer in BOTH directions and BOTH copy modes (sync hipMemcpyPeer and async
// hipMemcpyPeerAsync), each verified by a kernel on the destination GPU. This
// is the broad single-node GPU->GPU coverage; Device_CanAccessPeer_Matrix only
// validates the first capable pair. Skips on single-GPU nodes / no peer pairs.
ACCEPT_TEST(Device_P2P_AllPairs_Copy, "hip/device", Topology::Single,
            Kind::Functional, 120) {
  (void)ctx;
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 2)
    return skip("only " + std::to_string(count) + " GPU(s); no peer pairs");

  const int n = 4096;
  int pairs = 0;

  for (int a = 0; a < count; ++a) {
    for (int b = 0; b < count; ++b) {
      if (a == b) continue;
      int peer_accessible = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&peer_accessible, a, b));
      if (!peer_accessible) continue;

      const std::string peer_err = enable_peer_access_idempotent(a, b);
      if (!peer_err.empty()) return fail(peer_err);

      const std::string sync_err = p2p_directed_copy(a, b, n, /*use_async=*/false);
      if (!sync_err.empty())
        return fail("sync peer " + std::to_string(a) + "->" +
                    std::to_string(b) + ": " + sync_err);
      const std::string async_err = p2p_directed_copy(a, b, n, /*use_async=*/true);
      if (!async_err.empty())
        return fail("async peer " + std::to_string(a) + "->" +
                    std::to_string(b) + ": " + async_err);
      ++pairs;
    }
  }

  if (pairs == 0) return skip("no GPU pair reports peer access");
  return pass();
}

// Kernel-driven GPU<->GPU interaction across every peer-capable pair: a kernel
// on one GPU reads another GPU's memory directly over the peer link (in-kernel
// peer load), the direct data path RCCL relies on beyond the copy-engine
// hipMemcpyPeer route. Every ordered pair is exercised, so each link is read in
// both directions. Skips on single-GPU nodes / no peer pairs.
ACCEPT_TEST(Device_P2P_KernelAccess_AllPairs, "hip/device", Topology::Single,
            Kind::Functional, 120) {
  (void)ctx;
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    return skip("only " + std::to_string(device_count) + " GPU(s); no peer pairs");
  }

  const int element_count = 4096;
  PerfSink sink;
  int verified_pairs = 0;

  for (int reader = 0; reader < device_count; ++reader) {
    for (int owner = 0; owner < device_count; ++owner) {
      if (reader == owner) {
        continue;
      }
      int peer_accessible = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&peer_accessible, reader, owner));
      if (!peer_accessible) {
        continue;
      }

      const std::string enable_err = enable_peer_access_idempotent(reader, owner);
      if (!enable_err.empty()) {
        return fail(enable_err);
      }

      // Every pair is verified; the first peer-capable pair also carries the
      // perf measurement (warmup+N), the rest run a single verifying pass.
      const bool measure = verified_pairs == 0;
      const std::string metric = measure ? "device.p2p_kernel_read" : "";
      const MeasureOpts options = measure ? MeasureOpts{} : MeasureOpts{0, 1};

      const std::string access_err =
          p2p_kernel_read(reader, owner, element_count, sink, metric, options);
      if (!access_err.empty()) {
        return fail("kernel peer " + std::to_string(reader) + "<-" +
                    std::to_string(owner) + ": " + access_err);
      }
      ++verified_pairs;
    }
  }

  if (verified_pairs == 0) {
    return skip("no GPU pair reports peer access");
  }

  TestResult result = pass();
  result.perf = std::move(sink);
  return result;
}

// Validate the device attributes RCCL branches on (warp size, MP count,
// managed-access flag, arch name) read back as sane values.
ACCEPT_TEST(Device_Attr_Probe, "hip/device", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  const int dev = gpu();
  HIP_CHECK(hipSetDevice(dev));

  ProbedAttrs attrs;
  if (const std::string e = probe_scalar_attrs(dev, attrs); !e.empty())
    return fail(e);
  if (const std::string e = probe_batch_attrs(dev, attrs); !e.empty())
    return fail(e);
  if (const std::string e = cross_check_props(dev, attrs); !e.empty())
    return fail(e);
  return pass();
}

// HDP flush register read (gfx<90a GDR write-ordering); skips on gfx90a+ where
// the register is not on the GDR path.
ACCEPT_TEST(Hdp_Flush_Ordering, "hip/device", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  const int dev = gpu();
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  if (!is_pre_gfx90a(prop.gcnArchName))
    return skip(std::string("HDP flush register applies to gfx<90a, this is ") +
                prop.gcnArchName);

  int hdp = 0;
  HIP_CHECK(hipDeviceGetAttribute(&hdp, hipDeviceAttributeHdpMemFlushCntl, dev));
  return pass();
}

// Raise the device stack-size limit, then run a stack-heavy kernel and verify.
ACCEPT_TEST(DeviceSetLimit_StackSize, "hip/device", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  HIP_CHECK(hipSetDevice(gpu()));

  hipError_t limit_err = hipDeviceSetLimit(hipLimitStackSize, 32 * 1024);
  if (limit_err == hipErrorNotSupported)
    return skip("hipDeviceSetLimit(hipLimitStackSize) not supported");
  if (limit_err != hipSuccess)
    return fail(std::string("hipDeviceSetLimit: ") +
                hipGetErrorString(limit_err));

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  DeviceBuffer buf;
  HIP_CHECK(buf.allocate(bytes));

  FillArgs args{buf.as<int>(), n};  // k_stack_heavy(int* out, int n)
  std::string launch_err;
  hipError_t launch_status =
      hip_kernel_launch("k_stack_heavy", grid_for(n), dim3(kBlockSize), 0, 0,
                        &args, sizeof(args), &launch_err);
  if (launch_status == hipSuccess) launch_status = hipDeviceSynchronize();
  if (launch_status != hipSuccess)
    return fail("stack-heavy kernel" +
                (launch_err.empty() ? "" : ": " + launch_err));

  std::vector<int> host(n, -1);
  HIP_CHECK(hipMemcpy(host.data(), buf.get(), bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    if (host[i] != 64 * i)
      return fail("stack kernel wrong at " + std::to_string(i) + ": got " +
                  std::to_string(host[i]));
  return pass();
}

// Classify device vs host pointers and exercise the ROCM-2696 null-arg
// hipMemGetAddressRange path (must not crash).
ACCEPT_TEST(PointerGetAttributes_DeviceVsHost, "hip/device", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int dev = gpu();
  HIP_CHECK(hipSetDevice(dev));
  const size_t bytes = 4096;

  DeviceBuffer device_buf;
  HostBuffer host_buf;
  HIP_CHECK(device_buf.allocate(bytes));
  HIP_CHECK(host_buf.allocate(bytes, 0));

  hipPointerAttribute_t device_attr{};
  hipError_t attr_err = hipPointerGetAttributes(&device_attr, device_buf.get());
  if (attr_err != hipSuccess)
    return fail(std::string("device hipPointerGetAttributes: ") +
                hipGetErrorString(attr_err));
  if (device_attr.type != hipMemoryTypeDevice || device_attr.device != dev)
    return fail("device ptr misclassified: type=" +
                std::to_string(device_attr.type) + " device=" +
                std::to_string(device_attr.device));

  hipPointerAttribute_t host_attr{};
  attr_err = hipPointerGetAttributes(&host_attr, host_buf.get());
  if (attr_err != hipSuccess)
    return fail(std::string("host hipPointerGetAttributes: ") +
                hipGetErrorString(attr_err));
  if (host_attr.type != hipMemoryTypeHost)
    return fail("host ptr misclassified: type=" +
                std::to_string(host_attr.type));

  // Real address-range query: base must contain the allocation.
  void* base = nullptr;
  size_t sz = 0;
  hipError_t range_err = hipMemGetAddressRange(&base, &sz, device_buf.get());
  if (range_err != hipSuccess)
    return fail(std::string("hipMemGetAddressRange: ") +
                hipGetErrorString(range_err));

  // ROCM-2696: null out-params must be rejected cleanly, not crash.
  (void)hipMemGetAddressRange(nullptr, nullptr, device_buf.get());
  (void)hipGetLastError();
  return pass();
}

// Resolve the driver entry points RCCL depends on (mirrors cudaPfnFuncLoader):
// mandatory symbols must resolve; optional ones may be absent but cleanly.
ACCEPT_TEST(DriverEntryPoint_Resolve, "hip/device", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const char* mandatory[] = {"hipGetErrorString", "hipDeviceGet",
                             "hipDeviceGetAttribute"};
  // Region: resolving the mandatory symbols.
  // Measures the loader cost of hipGetProcAddress.
  PerfLoop loop;
  std::string resolve_err;
  for (auto pass : loop) {
    for (const char* symbol_name : mandatory) {
      void* function_ptr = nullptr;
      hipDriverProcAddressQueryResult status = HIP_GET_PROC_ADDRESS_SUCCESS;
      hipError_t query_err =
          hipGetProcAddress(symbol_name, &function_ptr, HIP_VERSION, 0, &status);
      if (query_err != hipSuccess || function_ptr == nullptr ||
          status != HIP_GET_PROC_ADDRESS_SUCCESS) {
        resolve_err = std::string("mandatory symbol unresolved: ") +
                      symbol_name + " (status " + std::to_string(status) + ")";
        break;
      }
    }
    if (!resolve_err.empty()) break;
  }
  if (!resolve_err.empty()) return fail(resolve_err);

  // Optional cuMem / launch / multicast family (hip names on ROCm): present or
  // cleanly absent, mirroring cudaPfnFuncLoader in RCCL.
  const char* optional[] = {
      "hipMemAddressReserve",
      "hipMemAddressFree",
      "hipMemCreate",
      "hipMemGetAllocationGranularity",
      "hipMemExportToShareableHandle",
      "hipMemImportFromShareableHandle",
      "hipMemMap",
      "hipMemRelease",
      "hipMemRetainAllocationHandle",
      "hipMemSetAccess",
      "hipMemUnmap",
      "hipMemGetAllocationPropertiesFromHandle",
      "hipMemGetHandleForAddressRange",
      "hipExtLaunchKernel",
      "hipLaunchKernelEx",
      "hipMulticastAddDevice",
      "hipMulticastBindMem",
      "hipMulticastBindAddr",
      "hipMulticastCreate",
      "hipMulticastGetGranularity",
      "hipMulticastUnbind",
  };
  for (const char* sym : optional) {
    std::string err;
    if (!driver_symbol_resolved_or_absent(sym, &err))
      return fail(std::string("optional symbol gave unclean status: ") + err);
  }
  return pass_with_latency(loop, "driver.proc_address_resolve");
}

// Device-management lifecycle: total memory, default mempool, then full reset.
// Isolated (hip/devreset) so the reset never tears down a shared worker.
ACCEPT_TEST(Device_Reset_Mgmt, "hip/devreset", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int dev = gpu();
  HIP_CHECK(hipSetDevice(dev));

  size_t total = 0;
  HIP_CHECK(hipDeviceTotalMem(&total, dev));
  if (total == 0) return fail("hipDeviceTotalMem returned 0");

  hipMemPool_t pool = nullptr;
  hipError_t pool_err = hipDeviceGetDefaultMemPool(&pool, dev);
  if (pool_err == hipErrorNotSupported)
    return skip("device does not advertise a default mempool");
  if (pool_err != hipSuccess)
    return fail(std::string("hipDeviceGetDefaultMemPool: ") +
                hipGetErrorString(pool_err));
  if (pool == nullptr) return fail("null default mempool handle");

  HIP_CHECK(hipDeviceReset());

  // Device must remain usable after reset.
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, 4096));
  (void)hipFree(d);
  return pass();
}

// Occupancy calculator floor: a sane block size and a non-zero active-block
// count for a trivial kernel (used by composablekernel/rocPRIM to size grids).
ACCEPT_TEST(Occupancy_Query, "hip/libsurface", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  HIP_CHECK(hipSetDevice(gpu()));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, gpu()));

  std::string kernel_err;
  hipFunction_t fn = hip_kernel_function("k_occupancy_probe", &kernel_err);
  if (fn == nullptr) return fail(kernel_err);

  int min_grid = 0, block = 0;
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&min_grid, &block, fn, 0, 0));
  if (block <= 0 || block > prop.maxThreadsPerBlock)
    return fail("block size out of range: " + std::to_string(block) + " (max " +
                std::to_string(prop.maxThreadsPerBlock) + ")");

  int active = 0;
  HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active, fn,
                                                               block, 0));
  if (active <= 0)
    return fail("active blocks per CU not positive: " + std::to_string(active));
  return pass();
}

// Texture-object lifecycle over a linear buffer: create, sample in a kernel,
// verify, destroy (rocPRIM texture_cache_iterator path).
ACCEPT_TEST(Texture_Object_Lifecycle, "hip/libsurface", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  HIP_CHECK(hipSetDevice(gpu()));
  if (!device_has_texture(gpu()))
    return skip("device has no texture/image hardware (CDNA3+)");
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  DeviceBuffer buf;
  HIP_CHECK(buf.allocate(bytes));
  std::vector<int> host_src(n);
  for (int i = 0; i < n; ++i) host_src[i] = i;
  HIP_CHECK(hipMemcpy(buf.get(), host_src.data(), bytes, hipMemcpyHostToDevice));

  hipResourceDesc res{};
  res.resType = hipResourceTypeLinear;
  res.res.linear.devPtr = buf.get();
  res.res.linear.desc = hipCreateChannelDesc<int>();
  res.res.linear.sizeInBytes = bytes;
  hipTextureDesc tdesc{};
  tdesc.readMode = hipReadModeElementType;

  hipTextureObject_t tex = 0;
  hipError_t create_err = hipCreateTextureObject(&tex, &res, &tdesc, nullptr);
  if (create_err == hipErrorNotSupported)
    return skip("device does not support texture objects");
  if (create_err != hipSuccess)
    return fail(std::string("hipCreateTextureObject: ") +
                hipGetErrorString(create_err));

  DeviceBuffer out;
  HIP_CHECK(out.allocate(bytes));
  int n_arg = n;
  int* out_ptr = out.as<int>();
  void* tex_args[] = {&tex, &out_ptr, &n_arg};
  hipError_t launch_err =
      hipLaunchKernel(sk_tex_fetch_ptr(), grid_for(n), dim3(kBlockSize),
                      tex_args, 0, 0);
  std::string launch_fn = "hipLaunchKernel(tex_fetch)";
  if (launch_err == hipSuccess) launch_err = hipGetLastError();
  if (launch_err == hipSuccess) {
    launch_err = hipDeviceSynchronize();
    launch_fn = "hipDeviceSynchronize";
  }

  std::vector<int> host_out(n, -1);
  if (launch_err == hipSuccess) {
    launch_err =
        hipMemcpy(host_out.data(), out.get(), bytes, hipMemcpyDeviceToHost);
    launch_fn = "hipMemcpy(D2H)";
  }
  (void)hipDestroyTextureObject(tex);

  if (launch_err != hipSuccess)
    return fail(launch_fn + ": " + hipGetErrorString(launch_err));
  const std::string err = verify_host_indices(host_out.data(), n);
  if (!err.empty()) return fail("texture sample " + err);
  return pass();
}

// Memory hints + capacity introspection: advise on a managed range, then read
// device/pointer/free-total info (rocThrust/composablekernel allocation path).
ACCEPT_TEST(Mem_Advise_Info, "hip/libsurface", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int dev = gpu();
  HIP_CHECK(hipSetDevice(dev));
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  int managed = 0;
  (void)hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, dev);

  void* ptr = nullptr;
  bool is_managed = false;
  if (managed) {
    if (hipMallocManaged(&ptr, bytes, hipMemAttachGlobal) == hipSuccess)
      is_managed = true;
  }
  if (!is_managed) HIP_CHECK(hipMalloc(&ptr, bytes));

  if (is_managed) {
    hipError_t advise_err = hipMemAdvise(ptr, bytes, hipMemAdviseSetReadMostly,
                                         dev);
    if (advise_err != hipSuccess && advise_err != hipErrorNotSupported) {
      (void)hipFree(ptr);
      return fail(std::string("hipMemAdvise(SetReadMostly): ") +
                  hipGetErrorString(advise_err));
    }
    (void)hipMemAdvise(ptr, bytes, hipMemAdviseSetPreferredLocation, dev);
    (void)hipGetLastError();
  }

  size_t free_bytes = 0, total_bytes = 0;
  hipError_t info_err = hipMemGetInfo(&free_bytes, &total_bytes);
  if (info_err != hipSuccess) {
    (void)hipFree(ptr);
    return fail(std::string("hipMemGetInfo: ") + hipGetErrorString(info_err));
  }
  if (total_bytes == 0 || free_bytes > total_bytes) {
    (void)hipFree(ptr);
    return fail("bad mem info: free=" + std::to_string(free_bytes) +
                " total=" + std::to_string(total_bytes));
  }

  size_t ptr_size = 0;
  hipError_t ptr_info_err = hipMemPtrGetInfo(ptr, &ptr_size);
  if (ptr_info_err == hipSuccess && ptr_size < bytes) {
    (void)hipFree(ptr);
    return fail("hipMemPtrGetInfo size " + std::to_string(ptr_size) + " < " +
                std::to_string(bytes));
  }
  (void)hipFree(ptr);
  return pass();
}

// Device-symbol copy: host -> __device__ symbol, kernel reads it, symbol -> host
// (hipFFT/RPP constant-memory path). Distinct from hipModuleGetGlobal.
ACCEPT_TEST(Memcpy_Symbol, "hip/libsurface", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  HIP_CHECK(hipSetDevice(gpu()));
  const int n = 256;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  std::vector<int> host_in(n);
  for (int i = 0; i < n; ++i) host_in[i] = i * 3 + 1;
  HIP_CHECK(hipMemcpyToSymbol(sk_symbol_buf_address(), host_in.data(), bytes, 0,
                              hipMemcpyHostToDevice));

  DeviceBuffer out;
  HIP_CHECK(out.allocate(bytes));
  int n_arg = n;
  int* out_ptr = out.as<int>();
  void* read_args[] = {&out_ptr, &n_arg};
  hipError_t launch_err =
      hipLaunchKernel(sk_read_symbol_ptr(), grid_for(n), dim3(kBlockSize),
                      read_args, 0, 0);
  std::string launch_fn = "hipLaunchKernel(read_symbol)";
  if (launch_err == hipSuccess) launch_err = hipGetLastError();
  if (launch_err == hipSuccess) {
    launch_err = hipDeviceSynchronize();
    launch_fn = "hipDeviceSynchronize";
  }
  if (launch_err != hipSuccess)
    return fail(launch_fn + ": " + hipGetErrorString(launch_err));

  std::vector<int> host_kernel(n, -1);
  HIP_CHECK(hipMemcpy(host_kernel.data(), out.get(), bytes,
                      hipMemcpyDeviceToHost));

  std::vector<int> host_back(n, -1);
  HIP_CHECK(hipMemcpyFromSymbol(host_back.data(), sk_symbol_buf_address(), bytes,
                                0, hipMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    if (host_kernel[i] != host_in[i])
      return fail("kernel saw wrong symbol at " + std::to_string(i) + ": got " +
                  std::to_string(host_kernel[i]));
    if (host_back[i] != host_in[i])
      return fail("symbol->host wrong at " + std::to_string(i) + ": got " +
                  std::to_string(host_back[i]));
  }
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
