/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <string>
#include <vector>

#include "wrapper/hip/api_wrapper.hpp"

namespace acceptance {
namespace hipw {

typedef struct hsa_lib_ops_t {
  decltype(::hsa_agent_get_info)* hsa_agent_get_info;
  decltype(::hsa_amd_agent_iterate_memory_pools)* hsa_amd_agent_iterate_memory_pools;
  decltype(::hsa_amd_memory_lock_to_pool)* hsa_amd_memory_lock_to_pool;
  decltype(::hsa_amd_memory_pool_get_info)* hsa_amd_memory_pool_get_info;
  decltype(::hsa_amd_memory_unlock)* hsa_amd_memory_unlock;
  decltype(::hsa_iterate_agents)* hsa_iterate_agents;
  // DMA-BUF export for GPU-Direct RDMA (verbs mixed tests). v2 adds the PCIE
  // mapping flag; close releases the fd. Absent on older ROCm -> MissingSymbol.
  decltype(::hsa_amd_portable_export_dmabuf)* hsa_amd_portable_export_dmabuf;
  decltype(::hsa_amd_portable_export_dmabuf_v2)* hsa_amd_portable_export_dmabuf_v2;
  decltype(::hsa_amd_portable_close_dmabuf)* hsa_amd_portable_close_dmabuf;
} hsa_lib_ops_t;

static std::vector<std::string> hsa_fn_names = {
    "hsa_agent_get_info",
    "hsa_amd_agent_iterate_memory_pools",
    "hsa_amd_memory_lock_to_pool",
    "hsa_amd_memory_pool_get_info",
    "hsa_amd_memory_unlock",
    "hsa_iterate_agents",
    "hsa_amd_portable_export_dmabuf",
    "hsa_amd_portable_export_dmabuf_v2",
    "hsa_amd_portable_close_dmabuf",
};

extern hsa_lib_ops_t hsa_lib_ops;

// Same call-site guard as HIP: a symbol absent on the running ROCm throws
// MissingSymbol (-> Skip at the boundary) instead of dereferencing null.
#define HIPW_HSA_OP(name) \
  (::acceptance::hipw::checked_op(::acceptance::hipw::hsa_lib_ops.name, #name))

#define hsa_agent_get_info HIPW_HSA_OP(hsa_agent_get_info)
#define hsa_amd_agent_iterate_memory_pools HIPW_HSA_OP(hsa_amd_agent_iterate_memory_pools)
#define hsa_amd_memory_lock_to_pool HIPW_HSA_OP(hsa_amd_memory_lock_to_pool)
#define hsa_amd_memory_pool_get_info HIPW_HSA_OP(hsa_amd_memory_pool_get_info)
#define hsa_amd_memory_unlock HIPW_HSA_OP(hsa_amd_memory_unlock)
#define hsa_iterate_agents HIPW_HSA_OP(hsa_iterate_agents)
#define hsa_amd_portable_export_dmabuf HIPW_HSA_OP(hsa_amd_portable_export_dmabuf)
#define hsa_amd_portable_export_dmabuf_v2 HIPW_HSA_OP(hsa_amd_portable_export_dmabuf_v2)
#define hsa_amd_portable_close_dmabuf HIPW_HSA_OP(hsa_amd_portable_close_dmabuf)

bool hsa_api_init();

void hsa_api_fini();

}  // namespace hipw
}  // namespace acceptance
