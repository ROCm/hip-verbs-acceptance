/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Runtime Module Loading group (MORI path).
//
// Built only when ACCEPT_WITH_HIP=ON. RCCL compiles kernels statically and never
// exercises hipModuleLoad; these tests cover the runtime code-object path that
// JIT/module consumers (e.g. MORI) depend on: hipModuleLoad(FILE) ->
// hipModuleGetFunction / hipModuleGetGlobal -> hipModuleLaunchKernel.
//
// The code object is produced at run time by hiprtc (HipModule.cpp) for the
// current device and written to a temp file, so there is no prebuilt artifact
// and no GPU arch baked in; the tests then load it from that path. When hiprtc is
// unavailable the tests skip.
//
// Conventions: deterministic functional checks -> MUST_PASS.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <cstddef>
#include <string>

#include <unistd.h>

#include "common/TestRegistry.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// Packed args for k_touch_global(int* out).
struct ModTouchArgs {
  int* out;
};

// RAII temp file holding the runtime-generated code object. Creates a unique
// path with mkstemp, fills it via hiprtc, and unlinks on destruction.
struct TempCodeObject {
  std::string path;
  bool ok = false;
  std::string err;

  TempCodeObject() {
    char tmpl[] = "/tmp/accept_co_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0) {
      err = "mkstemp failed";
      return;
    }
    ::close(fd);
    path = tmpl;
    ok = hip_kernels_write_code_object(path, &err);
    if (!ok) {
      ::unlink(path.c_str());
      path.clear();
    }
  }
  ~TempCodeObject() {
    if (!path.empty()) ::unlink(path.c_str());
  }
};

}  // namespace

// Load a code object from a FILE, resolve a kernel, launch, verify, unload cleanly.
ACCEPT_TEST(Module_Load_Launch_Lifecycle, "hip/module", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  const int n = 256;
  const unsigned block = 256;
  const unsigned grid = (n + block - 1) / block;

  TempCodeObject co;
  if (!co.ok) return fail("code object: " + co.err);

  hipModule_t mod = nullptr;
  hipError_t load_err = hipModuleLoad(&mod, co.path.c_str());
  if (load_err != hipSuccess)
    return fail("hipModuleLoad(" + co.path + "): " +
                hipGetErrorString(load_err));

  hipFunction_t fn = nullptr;
  hipError_t func_err = hipModuleGetFunction(&fn, mod, "k_fill_indices");
  if (func_err != hipSuccess) {
    (void)hipModuleUnload(mod);
    return fail(std::string("hipModuleGetFunction(k_fill_indices): ") +
                hipGetErrorString(func_err));
  }

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));

  FillArgs kargs{d_buf, n};
  size_t ksize = sizeof(kargs);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksize, HIP_LAUNCH_PARAM_END};
  hipError_t launch_err = hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0,
                                                s, nullptr, config);
  if (launch_err != hipSuccess) {
    (void)hipModuleUnload(mod);
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail(std::string("hipModuleLaunchKernel: ") +
                hipGetErrorString(launch_err));
  }
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  hipError_t unload_err = hipModuleUnload(mod);
  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("module launch result wrong at index " + std::to_string(bad_i) +
                ": got " + std::to_string(bad_v));
  if (unload_err != hipSuccess)
    return fail(std::string("hipModuleUnload: ") +
                hipGetErrorString(unload_err));
  HIP_CHECK(hipGetLastError());
  return pass();
}

// Resolve a module global, round-trip a value host/device, and confirm a kernel reads it.
ACCEPT_TEST(Module_GetGlobal_Resolve, "hip/module", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  TempCodeObject co;
  if (!co.ok) return fail("code object: " + co.err);

  hipModule_t mod = nullptr;
  hipError_t load_err = hipModuleLoad(&mod, co.path.c_str());
  if (load_err != hipSuccess)
    return fail("hipModuleLoad(" + co.path + "): " +
                hipGetErrorString(load_err));

  hipDeviceptr_t dptr = nullptr;
  size_t bytes = 0;
  hipError_t global_err = hipModuleGetGlobal(&dptr, &bytes, mod, "g_test_global");
  if (global_err != hipSuccess) {
    (void)hipModuleUnload(mod);
    return fail(std::string("hipModuleGetGlobal(g_test_global): ") +
                hipGetErrorString(global_err));
  }
  if (bytes < sizeof(int) || dptr == nullptr) {
    (void)hipModuleUnload(mod);
    return fail("hipModuleGetGlobal returned bad ptr/size (bytes=" +
                std::to_string(bytes) + ")");
  }

  // Host -> device write through the resolved global pointer, then read back.
  const int want = 12345;
  int got = 0;
  HIP_CHECK(hipMemcpy(dptr, &want, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(&got, dptr, sizeof(int), hipMemcpyDeviceToHost));
  if (got != want) {
    (void)hipModuleUnload(mod);
    return fail("global round-trip mismatch: got " + std::to_string(got) +
                ", expected " + std::to_string(want));
  }

  // Confirm the device actually observes the written global via a kernel.
  hipFunction_t fn = nullptr;
  hipError_t func_err = hipModuleGetFunction(&fn, mod, "k_touch_global");
  if (func_err != hipSuccess) {
    (void)hipModuleUnload(mod);
    return fail(std::string("hipModuleGetFunction(k_touch_global): ") +
                hipGetErrorString(func_err));
  }

  int* d_out = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
  HIP_CHECK(hipStreamCreate(&s));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(int)));

  ModTouchArgs kargs{d_out};
  size_t ksize = sizeof(kargs);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksize, HIP_LAUNCH_PARAM_END};
  hipError_t launch_err =
      hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, s, nullptr, config);
  if (launch_err != hipSuccess) {
    (void)hipModuleUnload(mod);
    (void)hipStreamDestroy(s);
    (void)hipFree(d_out);
    return fail(std::string("hipModuleLaunchKernel(k_touch_global): ") +
                hipGetErrorString(launch_err));
  }
  HIP_CHECK(hipStreamSynchronize(s));

  int kernel_saw = 0;
  HIP_CHECK(hipMemcpy(&kernel_saw, d_out, sizeof(int), hipMemcpyDeviceToHost));

  (void)hipModuleUnload(mod);
  (void)hipStreamDestroy(s);
  (void)hipFree(d_out);

  if (kernel_saw != want)
    return fail("kernel did not observe global: saw " +
                std::to_string(kernel_saw) + ", expected " +
                std::to_string(want));
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
