/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// HipKernels — the trivial __global__ device kernels shared by the HIP
// acceptance tests.
//
// These compile ONLY under a HIP device compiler (hipcc / clang -x hip), which
// defines __HIP__. In the login-node wrapper build (plain g++ + vendored
// headers, no HIP toolchain) __HIP__ is undefined and this header is empty, so
// host translation units that only need the host-side helpers in HipHelpers.hpp
// still compile. Kernel-launching tests are migrated to the module-launch path
// in a later step.

#if defined(ACCEPT_HAVE_HIP) && defined(__HIP__)

#include <hip/hip_runtime.h>

namespace acceptance {

// Single-element store/load kernels for cross-stream ordering checks.
// `static` gives each TU its own copy (no ODR clash if reused elsewhere).
// [[maybe_unused]]: this header is included by TUs that use only some kernels.
[[maybe_unused]] static __global__ void hip_write_flag(int* flag, int val) {
  *flag = val;
}
[[maybe_unused]] static __global__ void hip_read_flag(const int* flag, int* out) {
  *out = *flag;
}

// Empty kernel for timing / callback ordering tests.
[[maybe_unused]] static __global__ void hip_noop_kernel() {}

// Writes buf[i] = i for i < n. The canonical trivial kernel for launch tests.
[[maybe_unused]] static __global__ void hip_fill_indices(int* buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = i;
}

// Writes buf[i] = value for i < n. Used by the host-memory tests to prove the
// device actually reaches a mapped/registered host buffer through its device
// pointer.
[[maybe_unused]] static __global__ void hip_write_mapped_pattern(int* buf, int n,
                                                                 int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = value;
}

// Adds a constant: buf[i] += k for i < n. Used by the graph-capture tests to
// chain a second, order-dependent kernel after hip_fill_indices.
[[maybe_unused]] static __global__ void hip_add_constant(int* buf, int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += k;
}

// Device-side verify: atomically increments *mism for each i where buf[i] != i.
// Used by the IPC importer to prove the device can address the imported pointer.
[[maybe_unused]] static __global__ void hip_count_mismatch_indices(const int* buf,
                                                                   int n,
                                                                   int* mism) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && buf[i] != i) atomicAdd(mism, 1);
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP && __HIP__
