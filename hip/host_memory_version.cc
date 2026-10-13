/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Host Memory & Versioning / Symbol Detection group.
//
// Built only when ACCEPT_WITH_HIP=ON (LANGUAGE HIP). Covers the dependency-
// tracking floor (runtime/driver versions + optional-symbol presence) and the
// two host-memory paths RCCL relies on: mapped+uncached hipHostMalloc and
// hipHostRegister of existing host memory, each verified by a device write
// through the mapped device pointer.
//
// Conventions: deterministic functional checks (hard
// hipStreamSynchronize before every read-back) -> MUST_PASS. REGRESSION is
// reserved for crash/hang/timeout. Symbol_Version_Detection is informational:
// it reports present/absent per optional symbol and only fails if the HIP
// version queries themselves fail.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/TestRegistry.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"
#include "wrapper/SymbolProbe.hpp"

namespace acceptance {

namespace {

// One optional symbol to probe: which library to try and the symbol name.
struct OptionalSymbol {
  const char* lib;
  const char* name;
};

// Packed args for k_write_mapped_pattern(int* buf, int n, int value).
struct WritePatternArgs {
  int* buf;
  int n;
  int value;
};

}  // namespace

// Report HIP runtime/driver versions and present/absent status of optional dependency symbols.
ACCEPT_TEST(Symbol_Version_Detection, "hip/hostmem", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  int runtime_ver = 0;
  int driver_ver = 0;
  HIP_CHECK(hipRuntimeGetVersion(&runtime_ver));
  HIP_CHECK(hipDriverGetVersion(&driver_ver));

  // Optional symbols RCCL resolves at runtime. Expected on a ROCm stack: the
  // HSA dmabuf export pair; the cuMem family is NVIDIA-only and reports absent
  // on AMD; libionic (ionic direct-verbs) is present only on ionic clusters.
  const OptionalSymbol symbols[] = {
      {"libhsa-runtime64.so", "hsa_amd_portable_export_dmabuf"},
      {"libhsa-runtime64.so", "hsa_amd_portable_export_dmabuf_v2"},
      {"libcuda.so.1", "cuMemCreate"},
      {"libcuda.so.1", "cuMemGetHandleForAddressRange"},
      {"libcuda.so.1", "cuMemAddressReserve"},
      {"libcuda.so.1", "cuMemMap"},
      {"libcuda.so.1", "cuMemSetAccess"},
  };

  std::string report = "runtime=" + std::to_string(runtime_ver) +
                       " driver=" + std::to_string(driver_ver) + " |";
  for (const OptionalSymbol& s : symbols) {
    const bool present = hip_symbol_present(s.lib, s.name);
    report += " ";
    report += s.name;
    report += present ? "=present" : "=absent";
  }
  // ionic direct-verbs library: report library presence (no single canonical
  // entry point we depend on here).
  report += " libionic=";
  report += hip_lib_present("libionic.so") ? "present" : "absent";

  TestResult r;
  r.status = Status::Pass;
  r.reason = report;  // surfaces versions + per-symbol presence in report.json
  return r;
}

// Mapped uncached hipHostMalloc buffer: kernel writes via device pointer, host verifies.
ACCEPT_TEST(HostMalloc_Mapped_Uncached_DevPtr, "hip/hostmem", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = 256;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  const int pattern = 0x5a5a;

  int* h_ptr = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_ptr), bytes,
                          hipHostMallocMapped | hipHostMallocUncached));

  int* d_ptr = nullptr;
  hipError_t err =
      hipHostGetDevicePointer(reinterpret_cast<void**>(&d_ptr), h_ptr, 0);
  if (err != hipSuccess) {
    (void)hipHostFree(h_ptr);
    return fail(std::string("hipHostGetDevicePointer: ") +
                hipGetErrorString(err));
  }
  if (d_ptr == nullptr) {
    (void)hipHostFree(h_ptr);
    return fail("hipHostGetDevicePointer returned a null device pointer");
  }

  for (int i = 0; i < n; ++i) h_ptr[i] = 0;

  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));
  WritePatternArgs wargs{d_ptr, n, pattern};
  std::string launch_err;
  err = hip_kernel_launch("k_write_mapped_pattern", grid_for(n),
                          dim3(kBlockSize), 0, s, &wargs, sizeof(wargs),
                          &launch_err);
  if (err != hipSuccess) {
    (void)hipStreamDestroy(s);
    (void)hipHostFree(h_ptr);
    return fail(launch_err);
  }
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  for (int i = 0; i < n; ++i) {
    if (h_ptr[i] != pattern) {
      bad_i = i;
      bad_v = h_ptr[i];
      break;
    }
  }

  (void)hipStreamDestroy(s);
  (void)hipHostFree(h_ptr);

  if (bad_i >= 0)
    return fail("mapped host buffer wrong at index " + std::to_string(bad_i) +
                ": got " + std::to_string(bad_v) + ", expected " +
                std::to_string(pattern));
  return pass();
}

// Register host memory, kernel writes through the mapped device pointer, host verifies.
ACCEPT_TEST(HostRegister_Mapped_Uncached, "hip/hostmem", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = 256;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  const int pattern = 0x3c3c;

  void* raw = nullptr;
  if (::posix_memalign(&raw, 4096, bytes) != 0 || raw == nullptr)
    return fail("posix_memalign failed for " + std::to_string(bytes) +
                " bytes");
  int* buf = static_cast<int*>(raw);
  for (int i = 0; i < n; ++i) buf[i] = 0;

  hipError_t err =
      hipHostRegister(buf, bytes,
                      hipHostRegisterPortable | hipHostRegisterMapped |
                          hipExtHostRegisterUncached);
  if (err != hipSuccess) {
    ::free(raw);
    return fail(std::string("hipHostRegister: ") + hipGetErrorString(err));
  }

  int* d_ptr = nullptr;
  err = hipHostGetDevicePointer(reinterpret_cast<void**>(&d_ptr), buf, 0);
  if (err != hipSuccess) {
    (void)hipHostUnregister(buf);
    ::free(raw);
    return fail(std::string("hipHostGetDevicePointer: ") +
                hipGetErrorString(err));
  }
  if (d_ptr == nullptr) {
    (void)hipHostUnregister(buf);
    ::free(raw);
    return fail("hipHostGetDevicePointer returned a null device pointer");
  }

  hipStream_t s = nullptr;
  HIP_CHECK(hipStreamCreate(&s));
  WritePatternArgs wargs{d_ptr, n, pattern};
  std::string launch_err;
  err = hip_kernel_launch("k_write_mapped_pattern", grid_for(n),
                          dim3(kBlockSize), 0, s, &wargs, sizeof(wargs),
                          &launch_err);
  if (err != hipSuccess) {
    (void)hipStreamDestroy(s);
    (void)hipHostUnregister(buf);
    ::free(raw);
    return fail(launch_err);
  }
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  for (int i = 0; i < n; ++i) {
    if (buf[i] != pattern) {
      bad_i = i;
      bad_v = buf[i];
      break;
    }
  }

  (void)hipStreamDestroy(s);
  err = hipHostUnregister(buf);
  ::free(raw);

  if (bad_i >= 0)
    return fail("registered host buffer wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v) +
                ", expected " + std::to_string(pattern));
  if (err != hipSuccess)
    return fail(std::string("hipHostUnregister: ") + hipGetErrorString(err));
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
