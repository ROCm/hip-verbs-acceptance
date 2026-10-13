/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — HSA interop (NIC/GDR host-registration path).
//
// Built only when ACCEPT_WITH_HIP=ON; otherwise this file is empty. Exercises
// the low-level ionic/GDR path that locks host memory into a specific HSA pool
// for device/NIC DMA — it silently breaks on driver/FW/ROCr updates.

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

namespace acceptance {

namespace {

struct AgentSearch {
  hsa_agent_t gpu;
  bool gpu_found = false;
  hsa_agent_t cpu;
  bool cpu_found = false;
};

hsa_status_t collect_agents(hsa_agent_t a, void* data) {
  auto* s = static_cast<AgentSearch*>(data);
  hsa_device_type_t t{};
  if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (t == HSA_DEVICE_TYPE_GPU && !s->gpu_found) { s->gpu = a; s->gpu_found = true; }
  if (t == HSA_DEVICE_TYPE_CPU && !s->cpu_found) { s->cpu = a; s->cpu_found = true; }
  return HSA_STATUS_SUCCESS;
}

struct PoolSearch {
  hsa_amd_memory_pool_t pool;
  bool found = false;
};

hsa_status_t find_coarse_global_pool(hsa_amd_memory_pool_t p, void* data) {
  auto* s = static_cast<PoolSearch*>(data);
  hsa_amd_segment_t seg{};
  if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) !=
          HSA_STATUS_SUCCESS ||
      seg != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;
  uint32_t flags = 0;
  if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                   &flags) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (!(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED))
    return HSA_STATUS_SUCCESS;
  // Only an allocatable pool can back a host lock; aperture/doorbell pools are
  // coarse-grained too but reject lock_to_pool.
  bool alloc_ok = false;
  if (hsa_amd_memory_pool_get_info(
          p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_ok) !=
          HSA_STATUS_SUCCESS ||
      !alloc_ok)
    return HSA_STATUS_SUCCESS;
  s->pool = p;
  s->found = true;
  return HSA_STATUS_INFO_BREAK;
}

}  // namespace

// Lock host memory into a GPU agent's coarse-grained pool, write through the
// agent pointer with a kernel, and verify the host pages. SKIP if no pool.
ACCEPT_TEST(Hsa_MemoryPool_LockHostToPool, "hip/hsa", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  HIP_CHECK(hipSetDevice(0));  // ensures the HSA runtime is initialized

  AgentSearch as{};
  (void)hsa_iterate_agents(collect_agents, &as);
  if (!as.gpu_found) return skip("no HSA GPU agent found");
  if (!as.cpu_found) return skip("no HSA CPU agent found");

  // Host memory locks into a CPU/system coarse-grained pool; the GPU agent is
  // then granted access to that locked range.
  PoolSearch ps{};
  (void)hsa_amd_agent_iterate_memory_pools(as.cpu, find_coarse_global_pool, &ps);
  if (!ps.found) return skip("no allocatable coarse-grained system pool");

  const int n = 1024;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  int* host = static_cast<int*>(std::malloc(bytes));
  if (!host) return fail("host malloc failed");
  for (int i = 0; i < n; ++i) host[i] = -1;

  void* agent_ptr = nullptr;
  hsa_agent_t agents[1] = {as.gpu};
  hsa_status_t ls = hsa_amd_memory_lock_to_pool(host, bytes, agents, 1, ps.pool,
                                                /*flags=*/0, &agent_ptr);
  if (ls != HSA_STATUS_SUCCESS || agent_ptr == nullptr) {
    std::free(host);
    return skip("hsa_amd_memory_lock_to_pool unavailable (status " +
                std::to_string(ls) + ")");
  }

  FillArgs fa{static_cast<int*>(agent_ptr), n};
  std::string kerr;
  hipError_t launch_status = hip_kernel_launch("k_fill_indices", grid_for(n),
                                               dim3(kBlockSize), 0, 0, &fa,
                                               sizeof(fa), &kerr);
  if (launch_status == hipSuccess) launch_status = hipDeviceSynchronize();

  const std::string verify_err = (launch_status == hipSuccess)
                                     ? verify_host_indices(host, n)
                                     : std::string();
  (void)hsa_amd_memory_unlock(host);
  std::free(host);

  if (launch_status != hipSuccess)
    return fail("kernel via agent pointer" + (kerr.empty() ? "" : ": " + kerr));
  if (!verify_err.empty()) return fail("locked host buffer " + verify_err);
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
