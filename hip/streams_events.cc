/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Streams & Events group (first HIP module).
//
// Compiles as plain C++: HIP symbols resolve through the dlopen wrapper and
// kernels are built at run time by hiprtc, so the suite needs no HIP toolchain
// and runs on any arch. APIs that may be absent are declared optional in the
// wrapper and skip("requires <symbol>") at runtime. Built only when
// ACCEPT_WITH_HIP=ON.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <atomic>
#include <string>

#include "common/TestRegistry.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

// Create a non-blocking stream, query priority range and status, sync, destroy.
ACCEPT_TEST(Stream_Create_Flags_Lifecycle, "hip/streams", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  Stream stream;
  HIP_CHECK(stream.create_with_flags(hipStreamNonBlocking));

  // Priority range: HIP orders priorities numerically lowest = highest priority,
  // so the "least" (lo) must be >= the "greatest" (hi). Equal is valid (a device
  // that exposes a single priority level). Anything else is a bad range.
  int least = 0, greatest = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
  if (least < greatest)
    return fail("hipDeviceGetStreamPriorityRange inverted: least=" +
                std::to_string(least) + " greatest=" + std::to_string(greatest));

  // An empty stream should be either complete (Success) or legitimately not yet
  // ready; any other error is a failure.
  hipError_t query = hipStreamQuery(stream.get());
  if (query != hipSuccess && query != hipErrorNotReady)
    return fail(std::string("hipStreamQuery unexpected: ") +
                hipGetErrorString(query));

  HIP_CHECK(hipStreamSynchronize(stream.get()));
  return pass();
}

// Record a no-timing event on stream A; stream B waits and reads a flag written by A.
ACCEPT_TEST(Event_DisableTiming_CrossStream_Sync, "hip/streams",
            Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  Stream stream_writer, stream_reader;
  Event event;
  DeviceBuffer flag, out;
  HIP_CHECK(stream_writer.create());
  HIP_CHECK(stream_reader.create());
  HIP_CHECK(event.create_with_flags(hipEventDisableTiming));
  HIP_CHECK(flag.allocate(sizeof(int)));
  HIP_CHECK(out.allocate(sizeof(int)));
  HIP_CHECK(hipMemset(flag.get(), 0, sizeof(int)));
  HIP_CHECK(hipMemset(out.get(), 0, sizeof(int)));

  const int kSentinel = 42;  // value the writer stream stores into the flag
  // Writer stream: write sentinel -> record event. Args match k_write_flag(int*, int).
  struct WriteFlagArgs {
    int* flag;
    int val;
  } write_args{flag.as<int>(), kSentinel};
  HIP_KERNEL_LAUNCH("k_write_flag", dim3(1), dim3(1), 0, stream_writer.get(),
                    &write_args, sizeof(write_args));
  HIP_CHECK(hipEventRecord(event.get(), stream_writer.get()));

  // Reader stream: wait for event -> read flag. Args match k_read_flag(const int*, int*).
  HIP_CHECK(hipStreamWaitEvent(stream_reader.get(), event.get(), 0));
  struct ReadFlagArgs {
    const int* flag;
    int* out;
  } read_args{flag.as<int>(), out.as<int>()};
  HIP_KERNEL_LAUNCH("k_read_flag", dim3(1), dim3(1), 0, stream_reader.get(),
                    &read_args, sizeof(read_args));
  HIP_CHECK(hipStreamSynchronize(stream_reader.get()));

  int observed = -1;
  HIP_CHECK(hipMemcpy(&observed, out.get(), sizeof(int), hipMemcpyDeviceToHost));
  if (observed != kSentinel)
    return fail("cross-stream sync broken: got " + std::to_string(observed) +
                ", expected " + std::to_string(kSentinel));
  return pass();
}

// Record start/stop timing events around a kernel; elapsed time must be non-negative.
ACCEPT_TEST(Event_Timing_Query_Elapsed, "hip/streams", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  Event start, stop;
  Stream stream;
  HIP_CHECK(start.create());
  HIP_CHECK(stop.create());
  HIP_CHECK(stream.create());

  HIP_CHECK(hipEventRecord(start.get(), stream.get()));
  HIP_KERNEL_LAUNCH("k_noop", dim3(1), dim3(1), 0, stream.get(), nullptr, 0);
  HIP_CHECK(hipEventRecord(stop.get(), stream.get()));
  HIP_CHECK(hipEventSynchronize(stop.get()));

  // After synchronize, the stop event must report complete.
  HIP_CHECK(hipEventQuery(stop.get()));

  float elapsed_ms = -1.f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
  if (!(elapsed_ms >= 0.f))
    return fail("hipEventElapsedTime returned non-positive/NaN: " +
                std::to_string(elapsed_ms));
  return pass();
}

// Launch a kernel then a host callback on the same stream; callback must run once after the kernel.
ACCEPT_TEST(LaunchHostFunc_StreamCallback, "hip/streams", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  Stream stream;
  HIP_CHECK(stream.create());

  std::atomic<int> callback_count{0};
  HIP_KERNEL_LAUNCH("k_noop", dim3(1), dim3(1), 0, stream.get(), nullptr, 0);

  hipHostFn_t callback = [](void* arg) {
    static_cast<std::atomic<int>*>(arg)->fetch_add(1,
                                                    std::memory_order_relaxed);
  };
  HIP_CHECK(hipLaunchHostFunc(stream.get(), callback, &callback_count));
  HIP_CHECK(hipStreamSynchronize(stream.get()));

  const int count = callback_count.load(std::memory_order_relaxed);
  if (count != 1)
    return fail("callback ran " + std::to_string(count) + " times, expected 1");
  return pass();
}

// Swap thread capture mode with hipThreadExchangeStreamCaptureMode and restore the original mode.
ACCEPT_TEST(StreamCaptureMode_Exchange, "hip/streams", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;

  // 1) Read the current mode by swapping in Relaxed; `original` receives it.
  hipStreamCaptureMode original = hipStreamCaptureModeRelaxed;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&original));
  // Thread mode is now Relaxed; `original` holds whatever it was before.

  // 2) Swap in Global; the call must return the value we just set (Relaxed).
  hipStreamCaptureMode prev = hipStreamCaptureModeGlobal;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&prev));
  if (prev != hipStreamCaptureModeRelaxed) {
    // Best effort restore before failing.
    (void)hipThreadExchangeStreamCaptureMode(&original);
    return fail("swap did not return prior mode: got " +
                std::to_string(static_cast<int>(prev)) + ", expected Relaxed(" +
                std::to_string(static_cast<int>(hipStreamCaptureModeRelaxed)) +
                ")");
  }

  // 3) Restore the original mode (thread is currently Global).
  hipStreamCaptureMode restore = original;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&restore));
  if (restore != hipStreamCaptureModeGlobal)
    return fail("restore did not return Global as expected: got " +
                std::to_string(static_cast<int>(restore)));
  return pass();
}

// hipStreamGetId must return the same non-zero id for a stream across repeated queries.
ACCEPT_TEST(StreamGetId_Stable, "hip/streams", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  // If the running ROCm lacks hipStreamGetId the calls below trip the checked_op
  // guard (MissingSymbol -> Skip "requires hipStreamGetId").
  Stream stream;
  HIP_CHECK(stream.create());

  unsigned long long id1 = 0, id2 = 0;
  HIP_CHECK(hipStreamGetId(stream.get(), &id1));
  HIP_CHECK(hipStreamGetId(stream.get(), &id2));

  if (id1 == 0) return fail("hipStreamGetId returned 0 for a created stream");
  if (id1 != id2)
    return fail("stream id not stable: " + std::to_string(id1) + " vs " +
                std::to_string(id2));
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
