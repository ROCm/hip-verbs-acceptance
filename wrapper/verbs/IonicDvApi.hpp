/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Ionic Direct-Verbs (libionic) dynamic wrapper -- the vendor entry points
// RCCL's ionic transport uses on top of libibverbs (connect.cc):
//   * ionic_dv_pd_set_udma_mask(pd, mask) -- UDMA engine assignment, on the PD
//     before ibv_create_qp.
//   * ionic_dv_qp_set_gda(qp, en_send, en_recv) -- enable GPU-Direct-Async.
// plus OPTIONAL entry points that are not part of available():
//   * ionic_dv_pd_set_sqcmb(pd, enable, expdb, require) -- SQ placement
//     (controller memory buffer vs host memory) for QPs created on the PD.
//   * ionic_dv_is_ionic_ctx(ctx) -- true iff the context belongs to libionic
//     (the reliable "is this an ionic NIC" test: device names like rdma0 carry no
//     vendor prefix on renamed hosts).
//
// Mirrors RCCL's misc/ionicdvsymbols.cc: symbols resolved via dlvsym() at
// version "IONIC_1.0" (not plain dlsym), and the handle is opened RTLD_NODELETE
// and never dlclose'd -- libionic registers a driver into libibverbs on load, so
// unmapping it SIGSEGVs the next ibv_get_device_list (see common/Capability.cpp).
//
// No ionic header is vendored: the ABI and mask enum (LOW=1, HIGH=2) are
// reproduced from ionicdvcore.h. No-op unless ACCEPT_HAVE_IBVERBS.

#ifdef ACCEPT_HAVE_IBVERBS

#include <infiniband/verbs.h>

#include <cstdint>

namespace acceptance {
namespace ionicdv {

// Mirrors enum ionicdv_reg_udma_mask in RCCL's ionicdvcore.h.
enum UdmaMask : uint8_t {
  kUdmaLow  = 1,
  kUdmaHigh = 2,
};

// Exported libionic ABI (resolved via dlvsym @ IONIC_1.0). Both return 0 on
// success, non-zero errno on failure.
using pd_set_udma_mask_fn = int (*)(struct ibv_pd* pd, uint8_t udma_mask);
using qp_set_gda_fn       = int (*)(struct ibv_qp* qp, bool en_send, bool en_recv);
using pd_set_sqcmb_fn     = int (*)(struct ibv_pd* pd, bool enable, bool expdb,
                                    bool require);
using is_ionic_ctx_fn     = bool (*)(struct ibv_context* ctx);

// Load libionic and resolve both symbols. Idempotent; true iff both resolved.
// On failure the pointers stay null so dependent tests skip rather than fault.
bool init();
bool available();

// Resolved entry points, or null when unavailable.
pd_set_udma_mask_fn pd_set_udma_mask();
qp_set_gda_fn       qp_set_gda();

// Optional: ionic_dv_pd_set_sqcmb / ionic_dv_is_ionic_ctx, or null when libionic
// (or that symbol) is missing. Resolved by init() independently of the two
// required symbols, so they do NOT affect available(); call init() first, then
// null-check.
pd_set_sqcmb_fn     pd_set_sqcmb();
is_ionic_ctx_fn     is_ionic_ctx();

}  // namespace ionicdv
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
