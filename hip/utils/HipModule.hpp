/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// HipModule — runtime kernel provisioning for the HIP acceptance tests.
//
// The trivial test kernels are kept as HIP *source text* (see HipModule.cpp) and
// compiled for the CURRENT device at run time with hiprtc, then loaded with
// hipModuleLoadData. This means:
//   * no HIP device compiler is needed at build time (login-node build), and
//   * no GPU arch is baked into the binary -- the same binary runs on gfx942,
//     gfx950, or any future arch, because hiprtc targets whatever device is
//     present at run time.
//
// A test launches a kernel by name via hip_kernel_launch(); the module is
// compiled+loaded once per process on first use and cached. If hiprtc is not
// available on the runtime, hip_kernels_available() is false and callers should
// skip().
//
// Gated on ACCEPT_HAVE_HIP so the header is a no-op in non-HIP builds.

#ifdef ACCEPT_HAVE_HIP

#include "hip/utils/HipApi.hpp"

#include <cstddef>
#include <string>

namespace acceptance {

// True if runtime kernel compilation (hiprtc) is available. When false, kernel
// tests should skip("requires hiprtc") instead of failing.
bool hip_kernels_available();

// Launch test kernel `name` (an extern "C" __global__ from HipModule.cpp's
// source bundle) with a packed POD argument buffer. The shared module is
// compiled for the current device and loaded on first use (cached thereafter).
//
// `args`/`args_size` describe the kernel's packed argument struct, matching the
// device ABI (pointers 8B, ints 4B, natural alignment) -- the same convention as
// hipModuleLaunchKernel's HIP_LAUNCH_PARAM_BUFFER_POINTER path. Pass args=nullptr
// and args_size=0 for a no-argument kernel.
//
// On any failure (hiprtc missing, compile error, resolve/launch error) returns a
// non-success hipError_t and, when `err` is non-null, sets a descriptive reason
// (compile errors include the hiprtc build log).
hipError_t hip_kernel_launch(const char* name, dim3 grid, dim3 block,
                             unsigned shared_bytes, hipStream_t stream,
                             void* args, size_t args_size, std::string* err);

// Resolve test kernel `name` to its hipFunction_t (compiling the bundle on first
// use), for APIs that take a function handle directly -- e.g. the module
// occupancy calculator. Returns nullptr (with *err set) on any failure.
hipFunction_t hip_kernel_function(const char* name, std::string* err);

// Write the current-device code object (the hiprtc output for the shared kernel
// bundle) to `path`, compiling on first use. Lets a test exercise the
// hipModuleLoad(file) path with a runtime-generated .co -- no prebuilt artifact.
// Returns false (with *err set) if hiprtc is unavailable or the write fails.
bool hip_kernels_write_code_object(const std::string& path, std::string* err);

}  // namespace acceptance

// HIP_KERNEL_LAUNCH — launch a test kernel from inside an ACCEPT_TEST body; on
// any failure (compile/resolve/launch) return fail() with the reason. Mirrors
// HIP_CHECK's early-return convention (process isolation reclaims resources on
// the failure path). Guard kernel tests with hip_kernels_available() up front to
// skip cleanly when hiprtc is absent.
#define HIP_KERNEL_LAUNCH(name, grid, block, shared, stream, args, args_size)  \
  do {                                                                         \
    std::string _kl_err;                                                       \
    if (::acceptance::hip_kernel_launch(name, grid, block, shared, stream,     \
                                        args, args_size,                       \
                                        &_kl_err) != hipSuccess)               \
      return ::acceptance::fail(_kl_err);                                      \
  } while (0)

#endif  // ACCEPT_HAVE_HIP
