/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// static_kernels.cpp — device kernels that MUST be statically linked and
// registered (via __hipRegisterFunction) so the host-symbol launch APIs work:
// hipLaunchKernel(host)/hipGetFuncBySymbol(host)/hipLaunchKernelExC(host)/
// hipFuncSetAttribute(host). Those APIs resolve a HOST function pointer to a
// registered device kernel -- something a runtime-loaded module (hiprtc/.co)
// cannot provide.
//
// This TU is the ONLY part of the suite that needs a HIP device compiler. It is
// compiled ONCE by hipcc into a multi-arch (fat) relocatable object,
// prebuilt/kernels_static.o, which is vendored in the repo and linked into the
// login-node build with NO HIP toolchain present. Regenerate with:
//   hip/kernel_launch/build_static_kernels.sh
//
// extern "C" keeps the host-stub symbol names unmangled so the C++ test TU
// (compiled by g++) can take their addresses via matching plain declarations in
// StaticKernels.hpp.

#include <hip/hip_runtime.h>

// buf[i] = i for i < n. The canonical trivial launch kernel.
extern "C" __global__ void sk_fill_indices(int* buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = i;
}

// Device-side verify: atomically increments *mism for each i where buf[i] != i.
// Statically linked (not hiprtc) because the peer-copy test launches it on a
// SECOND device after hipSetDevice -- a runtime-loaded module is bound to one
// device's context, but a registered static kernel is available on every device.
extern "C" __global__ void sk_count_mismatch_indices(const int* buf, int n,
                                                     int* mism) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && buf[i] != i) atomicAdd(mism, 1);
}

// Stages buf[i] = i through dynamic shared memory so a launch must actually
// provision shared memory (drives the dyn-shmem opt-in / func-attribute tests).
extern "C" __global__ void sk_dyn_shmem_fill(int* buf, int n) {
  extern __shared__ int smem[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    smem[threadIdx.x] = i;
    __syncthreads();
    buf[i] = smem[threadIdx.x];
  }
}

// --- Statically-registered __device__ symbol + reader, for the constant-memory
// path (hipMemcpyToSymbol/FromSymbol via HIP_SYMBOL). That path resolves a
// registered host-shadow symbol, which only a statically linked __device__ var
// provides (hipModuleGetGlobal is the runtime-module equivalent, covered
// elsewhere).
extern "C" __device__ int g_symbol_buf[256] = {0};

extern "C" __global__ void sk_read_symbol(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = g_symbol_buf[i];
}

// The __device__ var's host-shadow symbol has internal linkage, so expose its
// registered address (what HIP_SYMBOL yields) for hipMemcpyTo/FromSymbol.
extern "C" const void* sk_symbol_buf_address() {
  return static_cast<const void*>(g_symbol_buf);
}

// --- Texture-object fetch. Texture intrinsics are a device-compiler feature, so
// this kernel must be built by hipcc (not hiprtc). On image-less archs (CDNA3+,
// gfx94x) __HIP_NO_IMAGE_SUPPORT is set and the fetch is compiled out; the test
// skips there at runtime.
extern "C" __global__ void sk_tex_fetch(hipTextureObject_t tex, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__HIP_NO_IMAGE_SUPPORT) && __HIP_NO_IMAGE_SUPPORT
  (void)tex;
  out[i] = -1;  // textures unavailable on this arch; test skips at runtime
#else
  out[i] = tex1Dfetch<int>(tex, i);
#endif
}
