/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// UdConn -- a minimal UD (Unreliable Datagram) verbs endpoint, the datagram
// counterpart to RcConn, backing the UD connection-baseline and 2-node recovery
// tests (RCCL's OOB / resiliency recovery runs over UD before the RC datapath).
//
// UD specifics this class encapsulates:
//   * IBV_QPT_UD; bring-up is INIT(QKEY)->RTR->RTS with no remote QPN/PSN/MTU.
//   * every SEND carries an ibv_ah + the peer's QPN/QKEY (wr.wr.ud) -- addressing
//     is per-WR, not per-QP.
//   * a UD receive prepends a 40-byte GRH, so recv buffers are payload+40 and the
//     payload starts at offset 40.
//
// GID resolution reuses RcConn's Capability probe. No-op unless ACCEPT_HAVE_IBVERBS.

#ifdef ACCEPT_HAVE_IBVERBS

#include "wrapper/verbs/IbvApi.hpp"

#include <cstdint>
#include <string>

namespace acceptance {
namespace verbsx {

// UD endpoint addressing swapped between peers over the bootstrap.
struct UdInfo {
  uint8_t  gid[16] = {};
  uint32_t qpn     = 0;
  uint32_t qkey    = 0;
  uint16_t lid     = 0;
};

// The GRH prefix bytes prepended to every UD receive.
constexpr uint32_t kUdGrhLen = 40;

class UdConn {
 public:
  UdConn() = default;
  ~UdConn();
  UdConn(const UdConn&) = delete;
  UdConn& operator=(const UdConn&) = delete;

  // Open an IB device and create PD/CQ/UD-QP up to INIT (QKEY set), filling
  // local(). dev_name empty -> ACCEPT_IB_DEVICE env var if set, else first
  // device with an ACTIVE port. Returns "" on success; sets no_device=true
  // when the host has no RDMA hardware (skip-worthy).
  std::string open(const std::string& dev_name, bool& no_device);

  // INIT -> RTR -> RTS. UD needs no remote addressing to reach RTS.
  std::string to_rts();

  // Recreate the UD QP from scratch (destroy + create + INIT). Models the UD
  // recovery path where the QP is torn down and a fresh QPN is published to the
  // peer. local().qpn is refreshed; call to_rts() again after this.
  std::string recreate();

  // Build an address handle for a peer from its UdInfo. Caller owns the returned
  // ibv_ah and must ibv_destroy_ah it (or use ibvw::safe_destroy_ah).
  struct ibv_ah* create_ah(const UdInfo& peer);

  // Post a UD SEND_WITH_IMM to `peer` via `ah`. The payload rides from a
  // registered MR (lkey/addr/len). imm is delivered in the receiver's WC.
  std::string post_send_imm(struct ibv_ah* ah, const UdInfo& peer, uint32_t lkey,
                            void* addr, uint32_t len, uint32_t imm,
                            uint64_t wr_id);

  // Post a UD receive. `len` is the PAYLOAD length; the buffer must be at least
  // len + kUdGrhLen bytes (the provider writes the 40-byte GRH first).
  std::string post_recv(uint32_t lkey, void* addr, uint32_t len, uint64_t wr_id);

  // Poll the CQ until a completion is reaped or timeout_ms elapses. Returns ""
  // on IBV_WC_SUCCESS; a reason on WC error / timeout / poll failure. out_wc, if
  // non-null, receives the reaped completion.
  std::string poll(int timeout_ms, struct ibv_wc* out_wc);

  struct ibv_pd*      pd()  const { return pd_; }
  struct ibv_context* ctx() const { return ctx_; }
  struct ibv_qp*      qp()  const { return qp_; }
  struct ibv_cq*      cq()  const { return cq_; }
  const UdInfo&       local() const { return local_; }
  const std::string&  dev_name() const { return dev_name_; }
  int                 gid_index() const { return gid_index_; }

 private:
  std::string create_qp_init();  // create UD QP + RESET->INIT (QKEY); fills local_

  struct ibv_context* ctx_ = nullptr;
  struct ibv_pd*      pd_   = nullptr;
  struct ibv_cq*      cq_   = nullptr;
  struct ibv_qp*      qp_   = nullptr;
  UdInfo              local_;
  std::string         dev_name_;
  int                 ib_port_   = 1;
  int                 gid_index_ = -1;  // -1 = no routable RoCEv2 GID
};

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
