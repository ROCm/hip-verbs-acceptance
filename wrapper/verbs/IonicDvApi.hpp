/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Ionic Direct-Verbs (libionic) dynamic wrapper -- the two vendor entry points
// RCCL's ionic transport uses on top of libibverbs (connect.cc):
//   * ionic_dv_pd_set_udma_mask(pd, mask) -- UDMA engine assignment, on the PD
//     before ibv_create_qp.
//   * ionic_dv_qp_set_gda(qp, en_send, en_recv) -- enable GPU-Direct-Async.
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

// Load libionic and resolve both symbols. Idempotent; true iff both resolved.
// On failure the pointers stay null so dependent tests skip rather than fault.
bool init();
bool available();

// Resolved entry points, or null when unavailable.
pd_set_udma_mask_fn pd_set_udma_mask();
qp_set_gda_fn       qp_set_gda();

}  // namespace ionicdv
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
