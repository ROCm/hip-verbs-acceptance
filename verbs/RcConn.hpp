/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// RcConn -- a minimal RC (Reliable Connection) verbs endpoint for the HIP+Verbs
// mixed data-path tests. Distilled from standard RC bring-up reference programs
// (ib_open / ib_connect / qp_to_rts), which all share the same RoCE bring-up.
//
// It owns one PD/CQ/QP and drives the RDMA data path (WRITE / WRITE_WITH_IMM /
// READ / ATOMIC_FETCH_AND_ADD) plus recv posting and CQ polling. Two wiring
// modes are supported:
//   * loopback  -- the QP is brought to RTS against its OWN QpInfo (self
//     connect), so a single process can issue RDMA to its own memory. Used by
//     the single-process tests.
//   * two-rank  -- QpInfo is swapped with the peer over common/Bootstrap and the
//     QP connects to the peer. Used by the TwoNode coherence test.
//
// Every ibv_* control-path call resolves through the dlopen wrapper
// (wrapper/verbs/IbvApi.hpp); the data-path calls (ibv_post_send/recv,
// ibv_poll_cq) are `static inline` in verbs.h dispatching through the provider
// ops table filled at ibv_open_device, so they are called directly.
//
// RoCE parameters: port 1, MTU from the port's active_mtu, is_global. The GID
// index is resolved at open() time to the routable RoCEv2 index for the
// chosen device (via Capability::detect(), which reads the sysfs gid_attrs
// types) rather than hardcoded, so cross-node connects use the correct IPv4
// RoCEv2 GID.
//
// The whole file is a no-op unless ACCEPT_HAVE_IBVERBS is defined.

#ifdef ACCEPT_HAVE_IBVERBS

#include "wrapper/verbs/IbvApi.hpp"

#include <cstdint>
#include <string>

namespace acceptance {
namespace verbsx {

// QP addressing info swapped between endpoints (or reused as-is for loopback).
struct QpInfo {
  uint8_t  gid[16] = {};
  uint32_t qpn     = 0;
  uint32_t psn     = 0;
  uint16_t lid     = 0;
};

// Remote memory window for an RDMA op target.
struct RemoteBuf {
  uint64_t addr = 0;
  uint32_t rkey = 0;
};

class RcConn {
 public:
  RcConn() = default;
  ~RcConn();
  RcConn(const RcConn&) = delete;
  RcConn& operator=(const RcConn&) = delete;

  // Open an IB device and create PD/CQ/QP up to state INIT, filling local().
  //   dev_name -- exact device name to open; empty -> the selected HCA
  //               (ACCEPT_HCA) if set, else first device with an ACTIVE port found by
  //               ibv_get_device_list (port 1 only; no multi-port scan).
  //   max_inline -- max_inline_data to request for the QP (0 = RCCL's default).
  //               The provider may grant more or less; see max_inline().
  // Returns "" on success, else a human-readable reason. Sets no_device=true
  // when the host simply has no RDMA hardware (skip-worthy vs. a real fault).
  std::string open(const std::string& dev_name, bool& no_device,
                   uint32_t max_inline = 0);

  // INIT -> RTR -> RTS against `remote`. For loopback pass local().
  std::string to_rts(const QpInfo& remote);

  // --- QP state control for the recovery/error tests ------------------------
  // Move the QP to ERR (outstanding WRs flush with IBV_WC_WR_FLUSH_ERR).
  std::string to_err();
  // Move the QP to RESET (drops all state; WQEs discarded).
  std::string to_reset();
  // RESET -> INIT, re-applying the access flags/port. After this, to_rts()
  // brings the same QP back up without destroying it. local() is refreshed.
  std::string reinit();

  // --- data path (each posts one signaled WR and, unless noted, does NOT poll)
  // remote_addr/rkey come from RemoteBuf; local_addr/lkey from the caller's MR.
  std::string post_write(uint32_t lkey, void* local_addr, uint32_t len,
                         const RemoteBuf& rb, uint64_t wr_id);
  std::string post_write_imm(uint32_t lkey, void* local_addr, uint32_t len,
                             const RemoteBuf& rb, uint32_t imm, uint64_t wr_id);
  std::string post_read(uint32_t lkey, void* local_addr, uint32_t len,
                        const RemoteBuf& rb, uint64_t wr_id);
  std::string post_atomic_fadd(uint32_t lkey, void* fetch_addr,
                               const RemoteBuf& rb, uint64_t add, uint64_t wr_id);
  std::string post_recv(uint32_t lkey, void* local_addr, uint32_t len,
                        uint64_t wr_id);

  // RDMA_WRITE carrying its payload inline (IBV_SEND_INLINE): no source MR/lkey,
  // the bytes ride in the WQE. len must fit the QP's max_inline_data.
  std::string post_write_inline(const void* local_addr, uint32_t len,
                                const RemoteBuf& rb, uint64_t wr_id);
  // Post a caller-built linked WR list (wr->next chain). The caller sets opcodes,
  // sg_lists, and signaling (typically only the last WR SIGNALED). This is a thin
  // pass-through to ibv_post_send for batched-submit tests.
  std::string post_chain(struct ibv_send_wr* first);

  // Poll the CQ until a completion is reaped or timeout_ms elapses. Returns ""
  // on a successful completion; a reason on WC error / timeout / poll error.
  // When out_wc is non-null it receives the reaped completion.
  std::string poll(int timeout_ms, struct ibv_wc* out_wc);

  struct ibv_pd*      pd()  const { return pd_; }
  struct ibv_context* ctx() const { return ctx_; }
  struct ibv_qp*      qp()  const { return qp_; }
  struct ibv_cq*      cq()  const { return cq_; }
  const QpInfo&       local() const { return local_; }
  const std::string&  dev_name() const { return dev_name_; }
  int                 gid_index() const { return gid_index_; }
  // max_inline_data the provider granted at ibv_create_qp (bytes).
  uint32_t            max_inline() const { return max_inline_; }

 private:
  std::string post_send_one(struct ibv_send_wr& wr);
  // Shared implementation for WRITE / WRITE_WITH_IMM / READ (Nikita refactor).
  std::string post_rdma(ibv_wr_opcode opcode, uint32_t lkey, void* local_addr,
                        uint32_t len, const RemoteBuf& rb, uint64_t wr_id,
                        uint32_t imm = 0);

  struct ibv_context* ctx_  = nullptr;
  struct ibv_pd*      pd_    = nullptr;
  struct ibv_cq*      cq_    = nullptr;
  struct ibv_qp*      qp_    = nullptr;
  QpInfo              local_;
  std::string         dev_name_;
  int                 ib_port_    = 1;
  int                 gid_index_  = -1;    // set by open(); -1 = no routable RoCEv2 GID
  bool                has_atomics_ = false; // set by open(); mirrors INIT access flags
  uint32_t            max_inline_  = 0;     // set by open(); granted max_inline_data
  enum ibv_mtu        path_mtu_   = IBV_MTU_4096; // see open()
};

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
