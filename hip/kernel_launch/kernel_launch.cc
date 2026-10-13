/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Kernel Launch group (RCCL launch+error path).
//
// Built only when ACCEPT_WITH_HIP=ON. The TU compiles as plain C++ (no HIP
// toolchain): HIP APIs resolve through the dlopen wrapper (HipApi.hpp). It covers
// the launch floor RCCL relies on before every collective:
//   * host-symbol launches -- hipLaunchKernel / hipGetFuncBySymbol /
//     hipLaunchKernelExC / hipFuncSetAttribute on a STATICALLY linked, runtime
//     -registered kernel (StaticKernels.hpp -> prebuilt kernels_static.o). These
//     APIs map a host function pointer to a device kernel, which a runtime-loaded
//     module cannot provide, so the kernels are statically linked.
//   * the module launch path -- hipModuleLaunchKernel on a function from a module
//     compiled at runtime by hiprtc (HipModule.hpp), needing no device compiler
//     at build time and no GPU arch baked in.
//   * hipLaunchCooperativeKernel on a stream whose idle hardware queue the
//     runtime reclaimed (ROCM-29369).
//
// Conventions: the tests are deterministic functional checks -> MUST_PASS, the
// crash test being the exception the policy classifies as KNOWN_FAILURE. The
// module-path legs skip() when hiprtc is unavailable on the runtime.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"
#include "hip/kernel_launch/StaticKernels.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// GPU_MAX_HW_QUEUES caps the hardware queues a process may hold; this is the
// runtime's own default when it is unset.
constexpr int kDefaultMaxHwQueues = 4;
// Queue holders the test keeps past that cap. The runtime takes an idle stream's
// queue back only while the holders outnumber the cap, and a wider margin loses
// more queues per cycle, so more streams can be the victim.
constexpr int kQueueHoldersOverCap = 5;
// A cap raised past this needs more streams than the test is willing to create.
constexpr int kMaxReclaimStreams = 64;
// A single cooperative launch races the reclaim window; a pair lands in it on
// every run.
constexpr int kCooperativeLaunchesPerStream = 2;
// The fault lands on the first cycle; the rest are margin for the idle order,
// which decides which stream loses its queue and varies per run.
constexpr int kReclaimCycles = 4;
// Long enough for the trivial kernels to drain while the host fences nothing.
// The runtime publishes no idle threshold, so this tracks observed behavior: a
// runtime that stops reclaiming here passes the test without arming the window.
constexpr int kIdleMs = 50;

// Whole-value integer parse, as elsewhere in the suite: "8junk" is not 8.
// Returns false when `text` is empty or not a plain integer.
bool parse_int(const char* text, int* value) {
  if (text == nullptr || *text == '\0') return false;
  try {
    const std::string number(text);
    std::size_t consumed = 0;
    const int parsed = std::stoi(number, &consumed);
    if (consumed != number.size()) return false;
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

// Streams sized to the caller's GPU_MAX_HW_QUEUES, so a raised cap widens the
// test instead of disarming it. Returns 0 with `skip_reason` set when the cap
// leaves nothing worth running.
int reclaim_stream_count(std::string* skip_reason) {
  int cap = kDefaultMaxHwQueues;
  const char* cap_env = std::getenv("GPU_MAX_HW_QUEUES");
  if (cap_env != nullptr && *cap_env != '\0' &&
      (!parse_int(cap_env, &cap) || cap <= 0)) {
    *skip_reason = std::string("GPU_MAX_HW_QUEUES=") + cap_env +
                   " is not a queue cap the test can size itself from";
    return 0;
  }
  const int streams = cap + kQueueHoldersOverCap - 1;
  if (streams > kMaxReclaimStreams) {
    *skip_reason = "GPU_MAX_HW_QUEUES needs " + std::to_string(streams) +
                   " streams to keep the queue reclaim armed";
    return 0;
  }
  return streams;
}

// The setting that turned the queue reclaim off process-wide, or "" while it is
// live, so a pass that never reached the window can say so. An unreadable value
// counts as off: the note only weakens a pass, never fails a run.
std::string queue_reclaim_disabled() {
  const char* dynamic_queues = std::getenv("DEBUG_HIP_DYNAMIC_QUEUES");
  if (dynamic_queues == nullptr || *dynamic_queues == '\0') return "";
  int enabled = 0;
  if (parse_int(dynamic_queues, &enabled) && enabled != 0) return "";
  return std::string("DEBUG_HIP_DYNAMIC_QUEUES=") + dynamic_queues;
}

}  // namespace

// Launch a kernel with dynamic shared memory (hipLaunchKernel on a static kernel)
// and confirm hipGetLastError is clean.
ACCEPT_TEST(Kernel_Launch_DynShared_LastError, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int n = 1024;
  const unsigned block = 256;
  const unsigned grid = (n + block - 1) / block;
  const size_t dyn = block * sizeof(int);

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernel(sk_dyn_shmem_fill_ptr(), dim3(grid), dim3(block),
                            args, dyn, s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("dyn-shmem launch result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  return pass();
}

// Raise the kernel dynamic shared-memory limit via hipFuncSetAttribute on a
// static kernel symbol, launch, and verify output.
ACCEPT_TEST(Launch_DynShared_FuncAttr, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));

  int optin = 0;
  HIP_CHECK(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin,
                                  dev));

  hipFuncAttributes fa{};
  HIP_CHECK(hipFuncGetAttributes(&fa, sk_dyn_shmem_fill_ptr()));

  const int n = 1024;
  const unsigned block = 256;
  const unsigned grid = (n + block - 1) / block;

  // Choose a dyn-shmem size to opt into: at least what the kernel needs, bounded
  // to a sane ceiling, and never above the device opt-in maximum.
  const size_t need = block * sizeof(int);
  size_t dyn = (optin > 0) ? static_cast<size_t>(optin) : need;
  dyn = std::min<size_t>(dyn, 32768);
  dyn = std::max<size_t>(dyn, need);

  HIP_CHECK(hipFuncSetAttribute(sk_dyn_shmem_fill_ptr(),
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(dyn)));

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernel(sk_dyn_shmem_fill_ptr(), dim3(grid), dim3(block),
                            args, dyn, s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("opt-in dyn-shmem launch wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v) +
                " (dyn=" + std::to_string(dyn) + ")");
  return pass();
}

// Smoke-test hipLaunchKernel on a static kernel (Part A) and hipModuleLaunchKernel
// on a hiprtc-compiled module function (Part B).
ACCEPT_TEST(Kernel_LaunchVariants_Smoke, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  const int n = 256;
  const dim3 grid(1), block(256);

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  // Poison the buffer so a no-op launch leaves detectable stale data (parity
  // with Part B, which memsets before its module launch).
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));

  // Part A: hipLaunchKernel with a void* args[] array on the static kernel.
  int n_arg = n;
  void* argsA[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernel(sk_fill_indices_ptr(), grid, block, argsA, 0, s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  if (!hip_verify_indices(d_buf, n, &bad_i, &bad_v)) {
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail("hipLaunchKernel result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  }

  // Part B: hipModuleLaunchKernel with packed args on a hiprtc module function.
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));
  FillArgs kargs{d_buf, n};
  std::string kerr;
  if (hip_kernel_launch("k_fill_indices", grid, block, 0, s, &kargs,
                        sizeof(kargs), &kerr) != hipSuccess) {
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail(kerr);
  }
  HIP_CHECK(hipStreamSynchronize(s));

  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("module launch result wrong at index " + std::to_string(bad_i) +
                ": got " + std::to_string(bad_v));
  return pass();
}

// Resolve a static device kernel with hipGetFuncBySymbol and launch it via
// hipModuleLaunchKernel.
ACCEPT_TEST(GetFuncBySymbol_Resolve, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int n = 256;
  const dim3 grid(1), block(256);

  hipFunction_t fn = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&fn, sk_fill_indices_ptr()));
  if (fn == nullptr)
    return fail("hipGetFuncBySymbol returned a null function");

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));

  FillArgs kargs{d_buf, n};
  size_t ksize = sizeof(kargs);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksize, HIP_LAUNCH_PARAM_END};
  hipError_t launch_err = hipModuleLaunchKernel(fn, grid.x, 1, 1, block.x, 1, 1,
                                                0, s, nullptr, config);
  if (launch_err != hipSuccess) {
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail(std::string("hipModuleLaunchKernel (resolved function): ") +
                hipGetErrorString(launch_err));
  }
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("resolved-function launch wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  return pass();
}

// Launch a single-block cooperative kernel through hipLaunchKernelExC with an
// attribute array on a static kernel symbol.
ACCEPT_TEST(LaunchKernelEx_ClusterAttrs, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int n = 256;

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  hipLaunchAttribute attrs[1];
  attrs[0].id = hipLaunchAttributeCooperative;
  attrs[0].val.cooperative = 1;

  hipLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(256);
  config.dynamicSmemBytes = 0;
  config.stream = s;
  config.attrs = attrs;
  config.numAttrs = 1;

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernelExC(&config, sk_fill_indices_ptr(), args));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("hipLaunchKernelExC result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  return pass();
}

// Two cooperative launches back to back on every stream, issued right after one
// stream was synchronized: the sequence that SIGSEGVs the HIP runtime in
// ROCM-29369.
ACCEPT_TEST(Launch_Cooperative_AfterQueueReclaim, "hip/launch", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  std::string skip_reason;
  const int stream_count = reclaim_stream_count(&skip_reason);
  if (stream_count == 0) return skip(skip_reason);
  const std::string reclaim_disabled = queue_reclaim_disabled();

  HIP_CHECK(hipSetDevice(0));
  int cooperative_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cooperative_supported,
                                  hipDeviceAttributeCooperativeLaunch, 0));
  if (cooperative_supported == 0)
    return skip("device does not support cooperative launch");

  const int n = 256;
  const dim3 grid(1), block(256);

  DeviceBuffer buf;
  HIP_CHECK(buf.allocate(sizeof(int) * n));
  int* d_buf = buf.as<int>();
  HIP_CHECK(hipMemset(d_buf, kPoisonByte, sizeof(int) * n));

  std::vector<Stream> streams(static_cast<std::size_t>(stream_count));
  for (Stream& stream : streams) HIP_CHECK(stream.create());

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  std::vector<double> burst_samples_us;
  for (int cycle = 0; cycle < kReclaimCycles; ++cycle) {
    for (Stream& stream : streams)
      HIP_CHECK(hipLaunchKernel(sk_fill_indices_ptr(), grid, block, args, 0,
                                stream.get()));

    // The queues have to go idle in hardware with nothing fenced; fencing here
    // (hipDeviceSynchronize) disarms the reclaim instead of arming it.
    std::this_thread::sleep_for(std::chrono::milliseconds(kIdleMs));

    // Synchronizing one stream is enough: it takes back the idle queues of all
    // the other streams as well.
    HIP_CHECK(hipStreamSynchronize(streams[0].get()));

    // Which stream lost its queue varies per run, so the burst goes to all of
    // them, and the cooperative launch must be the first operation on the stream.
    const PerfClock::time_point burst_start = PerfClock::now();
    for (Stream& stream : streams)
      for (int launch = 0; launch < kCooperativeLaunchesPerStream; ++launch)
        HIP_CHECK(hipLaunchCooperativeKernel(sk_fill_indices_ptr(), grid, block,
                                             args, 0, stream.get()));
    if (perf_probe_mode())
      burst_samples_us.push_back(std::chrono::duration<double, std::micro>(
                                     PerfClock::now() - burst_start).count());

    HIP_CHECK(hipDeviceSynchronize());
  }

  int bad_i = -1, bad_v = -1;
  if (!hip_verify_indices(d_buf, n, &bad_i, &bad_v))
    return fail("cooperative launch result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));

  PerfSink perf;
  emit_latency_samples(perf, "launch.cooperative_after_reclaim",
                       std::move(burst_samples_us));
  TestResult result = pass_with_perf(std::move(perf));
  result.reason = "streams=" + std::to_string(stream_count);
  if (!reclaim_disabled.empty())
    result.reason += ", " + reclaim_disabled +
                     " kept the queue reclaim off: this run could not reach the "
                     "ROCM-29369 window";
  return result;
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
