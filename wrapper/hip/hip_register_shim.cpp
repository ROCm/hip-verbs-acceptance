/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// hip_register_shim.cpp — definitions for the HIP fatbin/registration entry
// points referenced by the statically-linked kernels_static.o.
//
// The prebuilt object's load-time constructor registers its kernels with the
// runtime via __hipRegisterFatBinary/__hipRegisterFunction (and tears down via
// __hipUnregisterFatBinary). We deliberately do NOT link libamdhip64, so those
// symbols would be undefined at link time. These shims provide them: on first
// use each dlopen's libamdhip64 and forwards to the real entry point, so static
// registration reaches the runtime while the binary keeps zero ROCm link-time
// dependencies. __hipPush/PopCallConfiguration and hipLaunchKernel are also
// referenced by the device-stub code path and forwarded the same way.
//
// These are plain global C symbols (matching what hipcc-emitted code expects);
// the test TUs never see them -- they call HIP through the wrapper ops table.

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

namespace {

void* hip_runtime_handle() {
  static void* h = ::dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
  return h;
}

template <class Fn>
Fn resolve(const char* name) {
  return reinterpret_cast<Fn>(::dlsym(hip_runtime_handle(), name));
}

}  // namespace

extern "C" {

void** __hipRegisterFatBinary(const void* data) {
  using fn_t = void** (*)(const void*);
  static fn_t real = resolve<fn_t>("__hipRegisterFatBinary");
  return real ? real(data) : nullptr;
}

void __hipRegisterFunction(void** modules, const void* hostFunction,
                           char* deviceFunction, const char* deviceName,
                           unsigned int threadLimit, uint3* tid, uint3* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize) {
  using fn_t = void (*)(void**, const void*, char*, const char*, unsigned int,
                        uint3*, uint3*, dim3*, dim3*, int*);
  static fn_t real = resolve<fn_t>("__hipRegisterFunction");
  if (real)
    real(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid,
         bid, blockDim, gridDim, wSize);
}

void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar,
                      int ext, size_t size, int constant, int global) {
  using fn_t = void (*)(void**, void*, char*, char*, int, size_t, int, int);
  static fn_t real = resolve<fn_t>("__hipRegisterVar");
  if (real)
    real(modules, var, hostVar, deviceVar, ext, size, constant, global);
}

void __hipUnregisterFatBinary(void** modules) {
  using fn_t = void (*)(void**);
  static fn_t real = resolve<fn_t>("__hipUnregisterFatBinary");
  if (real) real(modules);
}

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, hipStream_t stream) {
  using fn_t = hipError_t (*)(dim3, dim3, size_t, hipStream_t);
  static fn_t real = resolve<fn_t>("__hipPushCallConfiguration");
  return real ? real(gridDim, blockDim, sharedMem, stream)
              : hipErrorSharedObjectSymbolNotFound;
}

hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                     size_t* sharedMem, hipStream_t* stream) {
  using fn_t = hipError_t (*)(dim3*, dim3*, size_t*, hipStream_t*);
  static fn_t real = resolve<fn_t>("__hipPopCallConfiguration");
  return real ? real(gridDim, blockDim, sharedMem, stream)
              : hipErrorSharedObjectSymbolNotFound;
}

// hipCreateChannelDesc(int,int,int,int,kind) is the extern "C" runtime symbol
// the header's hipCreateChannelDesc<T>() template specializations call. The
// texture test instantiates that template, so the symbol must exist at link
// time; forward it to the runtime like the registration entry points above (it
// is not in the ops table because the template names it directly).
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                          hipChannelFormatKind f) {
  using fn_t = hipChannelFormatDesc (*)(int, int, int, int, hipChannelFormatKind);
  static fn_t real = resolve<fn_t>("hipCreateChannelDesc");
  if (real) return real(x, y, z, w, f);
  return hipChannelFormatDesc{};
}

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks,
                           dim3 dimBlocks, void** args, size_t sharedMemBytes,
                           hipStream_t stream) {
  using fn_t = hipError_t (*)(const void*, dim3, dim3, void**, size_t,
                              hipStream_t);
  static fn_t real = resolve<fn_t>("hipLaunchKernel");
  return real ? real(function_address, numBlocks, dimBlocks, args,
                     sharedMemBytes, stream)
              : hipErrorSharedObjectSymbolNotFound;
}

}  // extern "C"
