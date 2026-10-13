/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// GdrMem -- HIP allocation modes x verbs registration methods, distilled from
// a HIP-allocation x verbs-registration reference program. Provides the 7
// allocation modes and 3 registration methods that program exercises, plus the
// host-side pattern fill/verify (via hipMemcpy) the mixed data-path tests reuse.
//
// Built only when BOTH HIP and ibverbs are enabled (ACCEPT_HAVE_HIP &&
// ACCEPT_HAVE_IBVERBS). Every HIP/HSA/ibv symbol resolves through the dlopen
// wrappers, so a runtime that lacks (e.g.) the dmabuf export entry points makes
// the dependent test Skip via MissingSymbol at the boundary.

#if defined(ACCEPT_HAVE_HIP) && defined(ACCEPT_HAVE_IBVERBS)

#include "hip/utils/HipApi.hpp"
#include "wrapper/verbs/IbvApi.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace acceptance {
namespace verbsx {

// 7 allocation modes (order + semantics match that reference program).
enum class Alloc {
  DeviceMalloc = 0,   // hipMalloc
  HostDefault,        // hipHostMalloc(Default)
  HostWC,             // hipHostMalloc(WriteCombined)
  HostCoherent,       // hipHostMalloc(Coherent)
  HostNonCoherent,    // hipHostMalloc(NonCoherent)
  FineGrain,          // hipExtMallocWithFlags(hipDeviceMallocFinegrained)
  Managed,            // hipMallocManaged(hipMemAttachGlobal)
  Count
};

// 3 registration methods.
enum class Reg {
  Std = 0,        // ibv_reg_mr (peer_mem / host pinned)
  DmabufV1,       // hsa_amd_portable_export_dmabuf     + ibv_reg_dmabuf_mr
  DmabufV2Pcie,   // hsa_amd_portable_export_dmabuf_v2  + ibv_reg_dmabuf_mr (PCIE)
  Count
};

const char* alloc_name(Alloc a);
const char* reg_name(Reg r);

// Access flags for an RC data-path MR: local write plus remote read/write, the
// bitmask the mixed and coherence tests all register their buffers with.
inline constexpr unsigned int kRdmaAccessFlags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

// A registered buffer: mr plus the dmabuf fd (>=0 only for the dmabuf paths).
struct Registration {
  struct ibv_mr* mr     = nullptr;
  int            dma_fd = -1;
};

// Allocate `size` bytes with `mode`. Returns nullptr on a (non-fatal) HIP
// allocation failure and, when err != nullptr, fills a reason. A missing HIP
// symbol throws MissingSymbol (-> Skip at the boundary).
void* alloc(Alloc mode, size_t size, std::string* err);
void  free_(Alloc mode, void* ptr);

// Register `ptr`/`size` in `pd` with `access` using `mode`. Returns a
// Registration with mr==nullptr on a (non-fatal) failure/unsupported path and,
// when err != nullptr, fills a reason (e.g. dmabuf export unsupported).
Registration reg(struct ibv_pd* pd, void* ptr, size_t size, unsigned int access,
                 Reg mode, std::string* err);
void         dereg(Registration& r);

// Host-visible pattern helpers (hipMemcpy H2D/D2H, like the reference). seed
// selects the byte ramp so stale data from a prior run is detectable.
bool fill_pattern(void* ptr, size_t size, uint8_t seed, std::string* err);
bool verify_pattern(const void* ptr, size_t size, uint8_t seed, std::string* err);

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP && ACCEPT_HAVE_IBVERBS
