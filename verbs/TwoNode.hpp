/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// TwoNode -- shared helpers for real cross-node (rank0<->rank1) verbs tests.
// RCCL's datapath is node-to-node RDMA; a loopback self-connect sidesteps real
// addressing, PSN handling, remote protection, and cross-node QP recovery, so
// the datapath/recovery tests connect two ranks over common/Bootstrap and drive
// RDMA between them. These helpers factor out the collective gate + connect dance
// (mirrors verbs/hip_alloc_matrix.cc) so each test body stays about its opcode.
//
// All tests using this are Topology::TwoNode and skip cleanly when run with a
// single rank (needs 2). Built only with ACCEPT_HAVE_IBVERBS.

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"
#include "verbs/UdConn.hpp"

#include <arpa/inet.h>  // ntohl
#include <time.h>       // nanosleep

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common/Bootstrap.hpp"
#include "common/Selection.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {
namespace verbsx {

// Fill `n` bytes at `p` with the deterministic seed pattern used by the verbs
// data-path tests, and its inverse check. mismatch() returns the first differing
// byte index, or -1 when every byte matches. Shared by data_path.cc /
// connection.cc so the fill/verify contract stays in one place.
inline void fill(uint8_t* p, size_t n, uint8_t seed) {
  for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>((seed + i) % 256);
}

inline long mismatch(const uint8_t* p, size_t n, uint8_t seed) {
  for (size_t i = 0; i < n; ++i)
    if (p[i] != static_cast<uint8_t>((seed + i) % 256)) return static_cast<long>(i);
  return -1;
}

// Outcome of reap_send_and_recv: OK on a verified recv, TIMEOUT when the recv
// did not arrive inside the budget (a dropped UD datagram -- retryable, not a
// hard error), FAILED on any hard error (poll error, WC error status, wrong
// opcode/imm, payload mismatch).
enum class Reap { OK, TIMEOUT, FAILED };

// Reap the send + recv completions for one exchange on `cq` and verify the recv.
// Both a send-side and a recv-side WR are expected (an RC/UD SEND lands as a
// recv). On the recv WC: require WITH_IMM == exp_imm, the payload at `payload`
// (len bytes, seed pattern) to match, and -- when exp_recv_wr_id != 0 -- the
// recv wr_id to match (0 disables the check).
//
// Returns OK / TIMEOUT / FAILED and, on a non-OK result, fills `reason`. The
// TIMEOUT/OK split lets UD callers retry a dropped datagram; callers on a
// reliable path treat TIMEOUT as a failure. This is the single reap-and-verify
// path shared by the SRQ lifecycle test and the 2-node UD recovery rounds.
inline Reap reap_send_and_recv(struct ibv_cq* cq, uint32_t exp_imm,
                               const uint8_t* payload, size_t len, uint8_t seed,
                               int timeout_ms, std::string* reason,
                               uint64_t exp_recv_wr_id = 0) {
  bool got_send = false, got_recv = false;
  long waited = 0;
  const long budget_us = static_cast<long>(timeout_ms) * 1000L;
  while (!got_send || !got_recv) {
    struct ibv_wc wc;
    std::memset(&wc, 0, sizeof(wc));
    const int n = ibv_poll_cq(cq, 1, &wc);
    if (n < 0) { *reason = "ibv_poll_cq failed"; return Reap::FAILED; }
    if (n == 0) {
      if (waited >= budget_us) return Reap::TIMEOUT;  // dropped datagram
      struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
      waited += 1000;
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      const char* s = ibv_wc_status_str(wc.status);
      *reason = std::string("completion error: ") + (s ? s : "unknown");
      return Reap::FAILED;
    }
    if (wc.opcode == IBV_WC_RECV) {
      got_recv = true;
      if (exp_recv_wr_id != 0 && wc.wr_id != exp_recv_wr_id) {
        *reason = "recv wr_id=" + std::to_string(wc.wr_id) + " (expected " +
                  std::to_string(exp_recv_wr_id) + ")";
        return Reap::FAILED;
      }
      if (!(wc.wc_flags & IBV_WC_WITH_IMM) || ntohl(wc.imm_data) != exp_imm) {
        *reason = "recv imm mismatch";
        return Reap::FAILED;
      }
      const long bad = mismatch(payload, len, seed);
      if (bad >= 0) {
        *reason = "payload mismatch at byte " + std::to_string(bad);
        return Reap::FAILED;
      }
    } else {
      got_send = true;
    }
  }
  return Reap::OK;
}

// Collective gate severity (0 ok < 1 skip < 2 fail): combine a per-rank outcome
// into a uniform decision so both ranks take the SAME branch and their collective
// sequences stay matched (an unmatched barrier/exchange deadlocks the pair).
enum Gate : uint8_t { G_OK = 0, G_SKIP = 1, G_FAIL = 2 };

inline Gate gate(Bootstrap* b, Gate mine) {
  assert(b->size() == 2);
  uint8_t s = static_cast<uint8_t>(mine);
  uint8_t all[2] = {0, 0};
  b->allgather(&s, all, 1);
  Gate g = G_OK;
  for (int i = 0; i < b->size(); ++i)
    g = std::max(g, static_cast<Gate>(all[i]));
  return g;
}

// If a NIC was selected (ACCEPT_HCA: --hca or GPU-NIC affinity), verify the
// QP actually opened on it. Guards against regressing the bug where --device
// (now --hca) only steered Capability's reported vendor/GID while RcConn::open()/
// UdConn::open() silently picked a different physical device for the
// connect itself. Returns "" when there's no request to check, or the
// request was honored; otherwise a reason.
inline std::string check_requested_device(const std::string& opened) {
  const std::string want = hca();
  if (want.empty() || opened == want) return "";
  return "connected on '" + opened + "' but the selected HCA is '" + want + "'";
}

// Open a device + RC QP and connect it to the peer (INIT->RTR->RTS), exchanging
// QpInfo over the bootstrap. On success `conn` is at RTS wired to the peer.
// Returns "" on success (both ranks connected); otherwise a reason, with
// *out_skip set when the collective decision was SKIP (no device / no peer).
inline std::string connect_peer(TestContext& ctx, RcConn& conn, bool* out_skip) {
  *out_skip = false;
  Bootstrap* boot = ctx.boot;

  // Phase 1: bring up the QP to INIT.
  Gate mine = G_OK;
  std::string reason;
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  if (!e.empty()) {
    mine = no_dev ? G_SKIP : G_FAIL;
    reason = e;
  } else if (conn.gid_index() < 0) {
    mine = G_SKIP;
    reason = "no routable RoCEv2 GID on " + conn.dev_name();
  } else if (std::string dr = check_requested_device(conn.dev_name());
             !dr.empty()) {
    mine = G_FAIL;
    reason = dr;
  }
  if (Gate g = gate(boot, mine); g != G_OK) {
    *out_skip = (g == G_SKIP);
    if (g == G_SKIP) return reason.empty() ? "peer has no RDMA device" : reason;
    return reason.empty() ? "peer QP open failed" : reason;
  }

  // Phase 2: exchange QpInfo with the peer and drive to RTS.
  QpInfo local = conn.local();
  QpInfo remote;
  boot->exchange(1 - ctx.rank, &local, &remote, sizeof(QpInfo));
  e = conn.to_rts(remote);
  Gate cm = e.empty() ? G_OK : G_FAIL;
  if (!e.empty()) reason = "connect: " + e;
  if (Gate g = gate(boot, cm); g != G_OK)
    return reason.empty() ? "peer connect failed" : reason;
  return "";
}

// Exchange a local RemoteBuf (addr+rkey of a registered MR) with the peer so
// each side knows where to target its RDMA ops.
inline RemoteBuf exchange_rbuf(TestContext& ctx, const RemoteBuf& mine) {
  RemoteBuf peer;
  ctx.boot->exchange(1 - ctx.rank, &mine, &peer, sizeof(RemoteBuf));
  return peer;
}

// ScopedMr -- RAII helper that factors out the repeated register/gate/exchange/
// dereg block common to every TwoNode datapath test:
//
//   ScopedMr mr(conn, buf.data(), size, access, boot, ctx);
//   if (!mr) return fail("ibv_reg_mr failed on a rank");
//   // ... use mr.lkey() for local ops, mr.peer_rb() for remote target ...
//
// On construction: registers the MR, does a collective gate so both ranks either
// proceed or both fail, exchanges RemoteBufs. If the gate fails, mr.ok()==false
// and the test body should return immediately. Destructor calls safe_dereg_mr.
class ScopedMr {
 public:
  ScopedMr(RcConn& conn, void* buf, size_t size, int access,
           Bootstrap* boot, TestContext& ctx)
      : mr_(ibv_reg_mr(conn.pd(), buf, size, access)) {
    Gate mine = mr_ ? G_OK : G_FAIL;
    ok_ = (gate(boot, mine) == G_OK);
    if (!ok_) { ibvw::safe_dereg_mr(mr_); mr_ = nullptr; return; }
    RemoteBuf mine_rb{reinterpret_cast<uint64_t>(buf),
                      mr_ ? mr_->rkey : 0u};
    peer_rb_ = exchange_rbuf(ctx, mine_rb);
  }
  ~ScopedMr() { ibvw::safe_dereg_mr(mr_); }
  ScopedMr(const ScopedMr&) = delete;
  ScopedMr& operator=(const ScopedMr&) = delete;

  explicit operator bool() const { return ok_; }
  bool ok() const { return ok_; }
  uint32_t lkey() const { return mr_ ? mr_->lkey : 0u; }
  const RemoteBuf& peer_rb() const { return peer_rb_; }

 private:
  struct ibv_mr* mr_ = nullptr;
  bool           ok_ = false;
  RemoteBuf      peer_rb_{};
};

// gate_return -- evaluate a per-rank outcome and return the collective verdict.
// Usage at the end of a TwoNode test body:
//   return gate_return(boot, reason);
// Returns pass() if all ranks succeed, fail(reason) otherwise.
inline TestResult gate_return(Bootstrap* boot, const std::string& reason) {
  Gate g = gate(boot, reason.empty() ? G_OK : G_FAIL);
  if (g == G_OK) return pass();
  return fail(reason.empty() ? "peer failed" : reason);
}

// UD counterpart to connect_peer: bring up a UD endpoint on both ranks, exchange
// UdInfo, and drive to RTS. On success `conn` is at RTS and `peer` holds the
// peer's UdInfo (gid/qpn/qkey/lid). Returns "" on success; otherwise a reason,
// with *out_skip set when the collective decision was SKIP (no device / no GID).
inline std::string ud_connect_peer(TestContext& ctx, UdConn& conn, UdInfo* peer,
                                   bool* out_skip) {
  *out_skip = false;
  Bootstrap* boot = ctx.boot;

  Gate mine = G_OK;
  std::string reason;
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  if (!e.empty()) {
    mine = no_dev ? G_SKIP : G_FAIL;
    reason = e;
  } else if (conn.gid_index() < 0) {
    mine = G_SKIP;
    reason = "no routable RoCEv2 GID on " + conn.dev_name();
  } else if (std::string dr = check_requested_device(conn.dev_name());
             !dr.empty()) {
    mine = G_FAIL;
    reason = dr;
  }
  if (Gate g = gate(boot, mine); g != G_OK) {
    *out_skip = (g == G_SKIP);
    if (g == G_SKIP) return reason.empty() ? "peer has no RDMA device" : reason;
    return reason.empty() ? "peer UD QP open failed" : reason;
  }

  UdInfo local = conn.local();
  boot->exchange(1 - ctx.rank, &local, peer, sizeof(UdInfo));
  e = conn.to_rts();
  Gate cm = e.empty() ? G_OK : G_FAIL;
  if (!e.empty()) reason = "UD RTS: " + e;
  if (gate(boot, cm) != G_OK)
    return reason.empty() ? "peer UD RTS failed" : reason;
  return "";
}

// Drive a single RC QP to RTS in loopback (self-connect): INIT -> RTR (dest_qp =
// its own QPN, RoCE GID-addressed via `gid`/`sgid`) -> RTS. Access flags are
// LOCAL_WRITE|REMOTE_WRITE (enough for a loopback SEND/WRITE target). Used by the
// single-node SRQ and ionic tests, which build a bespoke QP rather than RcConn's.
// Returns "" on success or a human-readable reason.
inline std::string rc_loopback_to_rts(struct ibv_qp* qp,
                                      const union ibv_gid& gid, int sgid) {
  struct ibv_qp_attr a;
  std::memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = 1;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                IBV_QP_ACCESS_FLAGS))
    return "INIT failed: " + std::string(std::strerror(errno));
  // Use the port's negotiated MTU, not a fixed guess (see RcConn::open()):
  // requesting a path_mtu the port never advertised fails RTR with EINVAL.
  struct ibv_port_attr pa;
  std::memset(&pa, 0, sizeof(pa));
  if (ibv_query_port(qp->context, 1, &pa))
    return "ibv_query_port failed: " + std::string(std::strerror(errno));
  std::memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = pa.active_mtu;
  a.dest_qp_num = qp->qp_num;  // loopback: connect to self
  a.rq_psn = 0;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  a.ah_attr.grh.dgid = gid;
  a.ah_attr.grh.sgid_index = static_cast<uint8_t>(sgid);
  a.ah_attr.grh.hop_limit = 64;
  a.ah_attr.port_num = 1;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    return "RTR failed: " + std::string(std::strerror(errno));
  std::memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.sq_psn = 0;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.max_rd_atomic = 1;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                                IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                IBV_QP_MAX_QP_RD_ATOMIC))
    return "RTS failed: " + std::string(std::strerror(errno));
  return "";
}

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
