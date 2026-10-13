/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// StaticKernels — g++-visible handles for the statically-linked device kernels
// defined in static_kernels.cpp (prebuilt into kernels_static.o by hipcc).
//
// hipcc emits each __global__ kernel's host-shadow symbol as a DATA object whose
// ADDRESS is the host function pointer that __hipRegisterFunction registers and
// that the host-symbol launch APIs accept (hipLaunchKernel / hipGetFuncBySymbol
// / hipLaunchKernelExC / hipFuncSetAttribute / hipFuncGetAttributes). We declare
// those symbols as byte arrays so g++ can take their address without a
// function-type mismatch, and expose const void* accessors for the tests.
//
// Gated on ACCEPT_HAVE_HIP so the header is a no-op in non-HIP builds.

#ifdef ACCEPT_HAVE_HIP

extern "C" {
extern char sk_fill_indices[];      // __global__ buf[i] = i
extern char sk_dyn_shmem_fill[];    // __global__ buf[i] = i via dynamic shmem
extern char sk_count_mismatch_indices[];  // __global__ count buf[i] != i
extern char sk_read_symbol[];       // __global__ out[i] = g_symbol_buf[i]
extern char sk_tex_fetch[];         // __global__ out[i] = tex1Dfetch<int>(tex,i)
// Registered host-shadow address of the __device__ g_symbol_buf[256] global
// (what HIP_SYMBOL yields), for hipMemcpyTo/FromSymbol.
const void* sk_symbol_buf_address();
}

namespace acceptance {

// Host function pointers for the host-symbol launch APIs.
inline const void* sk_fill_indices_ptr() {
  return reinterpret_cast<const void*>(sk_fill_indices);
}
inline const void* sk_dyn_shmem_fill_ptr() {
  return reinterpret_cast<const void*>(sk_dyn_shmem_fill);
}
inline const void* sk_count_mismatch_indices_ptr() {
  return reinterpret_cast<const void*>(sk_count_mismatch_indices);
}
inline const void* sk_read_symbol_ptr() {
  return reinterpret_cast<const void*>(sk_read_symbol);
}
inline const void* sk_tex_fetch_ptr() {
  return reinterpret_cast<const void*>(sk_tex_fetch);
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP
