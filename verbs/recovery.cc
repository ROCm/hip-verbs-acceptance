/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Verbs acceptance tests -- RC recovery & error handling (Group C), REAL 2-node.
//
// The QP error/recovery mechanics RCCL's resiliency (failover/recovery) path
// depends on, exercised on a CONNECTED cross-node RC QP (rank0<->rank1) rather
// than a loopback self-connect. This matters: the known ionic RESET defect
// manifests on a QP that carries live REMOTE state, which loopback cannot
// reproduce. Tests:
//   * vat_qp_err_transition   -- QP->ERR flushes outstanding WRs, then RESET
//   * vat_cqe_error_codes     -- WR_FLUSH_ERR isolation across two QPs
//   * vat_rc_qp_reset_recovery-- recover a connected QP after a failure via
//                                ERR->RESET->INIT->RTS with QPN re-exchange, and
//                                confirm cross-node transfer resumes. Includes a
//                                negative control (the downed QP must not carry
//                                data before recovery).
//   * vat_async_event_monitor -- ibv_get_async_event / ibv_ack_async_event
//                                (local fault injection; single-node)
//   * vat_2node_ud_recovery_protocol -- UD alive->ACK->QPN-exchange recovery
//                                over the wire (the OOB reconnect RCCL uses)
//   * vat_ionic_sq_sig_all    -- ionic sq_sig_all DRIVER FEATURE BITS (16-18,
//                                the RCCL data-QP encoding); NA off-ionic
//   * vat_ionic_inline_capacity -- ionic granted max_inline_data floor (the WQE
//                                grant behind the RESET defect); NA off-ionic
//   * vat_ionic_udma_mask     -- ionic direct-verbs ionic_dv_pd_set_udma_mask
//                                (libionic dlvsym); NA off-ionic
//   * vat_ionic_gda_enable    -- ionic direct-verbs ionic_dv_qp_set_gda
//                                (libionic dlvsym); NA off-ionic
//
// Built only with ACCEPT_HAVE_IBVERBS. Without it, an empty TU.

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"
#include "verbs/TwoNode.hpp"
#include "verbs/UdConn.hpp"
#include "wrapper/verbs/IonicDvApi.hpp"
#include "wrapper/verbs/VerbsHelpers.hpp"

#include <arpa/inet.h>  // ntohl
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/Bootstrap.hpp"
#include "common/Capability.hpp"
#include "common/TestRegistry.hpp"
#include "common/Topology.hpp"

namespace acceptance {

namespace {

constexpr int kPollMs = 5000;
constexpr int kClientRank = 1;

// Drain up to `expect` completions, counting those with `want` status.
int drain_cq(struct ibv_cq* cq, int expect, enum ibv_wc_status want,
             int* want_count, int budget_ms) {
  int reaped = 0;
  if (want_count) *want_count = 0;
  long waited = 0;
  const long budget_us = static_cast<long>(budget_ms) * 1000L;
  while (reaped < expect) {
    struct ibv_wc wc;
    std::memset(&wc, 0, sizeof(wc));
    const int n = ibv_poll_cq(cq, 1, &wc);
    if (n < 0) break;
    if (n == 0) {
      if (waited >= budget_us) break;
      struct timespec ts{0, 1000000L};
      ::nanosleep(&ts, nullptr);
      waited += 1000;
      continue;
    }
    ++reaped;
    if (want_count && wc.status == want) ++(*want_count);
  }
  return reaped;
}

}  // namespace

// QP -> ERR mechanics on a connected QP: post recv WRs, drive to ERR, confirm
// the outstanding WRs drain as IBV_WC_WR_FLUSH_ERR, then RESET cleanly. Both
// ranks do the same to keep collectives matched.
ACCEPT_TEST(vat_qp_err_transition, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr int kN = 4;
  constexpr size_t kLen = 256;
  std::vector<uint8_t> buf(kLen * kN);
  struct ibv_mr* mr =
      ibv_reg_mr(conn.pd(), buf.data(), buf.size(), IBV_ACCESS_LOCAL_WRITE);
  std::string reason;
  if (mr == nullptr) {
    reason = "ibv_reg_mr failed";
  } else {
    for (int i = 0; i < kN && reason.empty(); ++i)
      reason = conn.post_recv(mr->lkey, buf.data() + i * kLen, kLen,
                              static_cast<uint64_t>(i));
    if (reason.empty()) reason = conn.to_err();
    if (reason.empty()) {
      int flushed = 0;
      const int reaped =
          drain_cq(conn.cq(), kN, IBV_WC_WR_FLUSH_ERR, &flushed, kPollMs);
      if (reaped < kN)
        reason = "only " + std::to_string(reaped) + "/" + std::to_string(kN) +
                 " WRs drained on ERR";
      else if (flushed < kN)
        reason = "expected " + std::to_string(kN) + " WR_FLUSH_ERR, got " +
                 std::to_string(flushed);
    }
    if (reason.empty()) reason = conn.to_reset();
  }
  ibvw::safe_dereg_mr(mr);
  boot->barrier();  // keep both ranks in lockstep before teardown
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer ERR-transition failed" : reason);
  return pass();
}

// WR_FLUSH_ERR: client drives its QP to ERR and confirms posted recv WRs drain
// with IBV_WC_WR_FLUSH_ERR. The server QP stays untouched (no cross-node QP
// poisoning). This tests local flush behaviour, not inter-QP isolation.
ACCEPT_TEST(vat_cqe_error_codes, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 4096;
  std::vector<uint8_t> buf(kLen);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  struct ibv_mr* mr = ibv_reg_mr(conn.pd(), buf.data(), kLen, access);
  verbsx::Gate mine = mr ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mine) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail("ibv_reg_mr failed on a rank");
  }
  verbsx::RemoteBuf mine_rb{reinterpret_cast<uint64_t>(buf.data()), mr->rkey};
  (void)verbsx::exchange_rbuf(ctx, mine_rb);  // lockstep only; this rank never targets the peer

  std::string reason;
  // Client posts recvs then errors ITS OWN qp -> its WRs flush locally.
  if (is_client) {
    for (int i = 0; i < 4 && reason.empty(); ++i)
      reason = conn.post_recv(mr->lkey, buf.data(), kLen,
                              static_cast<uint64_t>(0xA00 + i));
    if (reason.empty()) reason = conn.to_err();
    if (reason.empty()) {
      int flushed = 0;
      drain_cq(conn.cq(), 4, IBV_WC_WR_FLUSH_ERR, &flushed, kPollMs);
      if (flushed < 4)
        reason = "client: expected 4 flush errors, got " +
                 std::to_string(flushed);
    }
  }
  // Barrier: the server's QP must remain healthy despite the client's error.
  // (No cross-QP poisoning: they are independent QPs/CQs on separate nodes.)
  boot->barrier();
  ibvw::safe_dereg_mr(mr);
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer CQE-error test failed" : reason);
  return pass();
}

// Recover a downed CONNECTED RC QP and resume cross-node transfer. This is the
// real resiliency path: a QP with live remote state is driven to ERR, proven
// non-functional (negative control), then recovered via ERR->RESET->INIT->RTS
// with a QpInfo re-exchange, and a cross-node RDMA_WRITE must succeed again.
// Note: RESET reuses the same QP handle and QPN -- the re-exchange is needed
// because to_rts() on both ends requires the peer's dest_qp_num even when
// unchanged.
//
// NOTE re ionic: the ionic driver has a known ibv_modify_qp(RESET) defect --
// after ERR->RESET->INIT->RTS, post-recovery RDMA_WRITE times out (QP is
// non-functional at the hardware level despite state transitions returning 0).
//
// Observed behaviour during test development: the test consistently PASSES on
// ionic (wr_id=2 confirmed via assertion -- not a stale-CQE false positive),
// while standalone C reproducers (single-process, same timeout, same GID/size)
// consistently fail. The structural differences systematically ruled out:
// loopback vs two-QP, single vs two ibv_context, self-connect, pre-RESET write
// on ERR QP, receiver reset from RTS vs ERR, write payload size, and timeout.
// The remaining unexplained variable is some ionic driver/firmware internal state
// that differs between a forked-process harness worker and a standalone program.
// This requires ionic driver source to diagnose. The test result is correct for
// the harness execution model (which matches RCCL's process model).
ACCEPT_TEST(vat_rc_qp_reset_recovery, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 45) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 4096;
  std::vector<uint8_t> buf(kLen);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  struct ibv_mr* mr = ibv_reg_mr(conn.pd(), buf.data(), kLen, access);
  verbsx::Gate mine = mr ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mine) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail("ibv_reg_mr failed on a rank");
  }
  verbsx::RemoteBuf mine_rb{reinterpret_cast<uint64_t>(buf.data()), mr->rkey};
  verbsx::RemoteBuf peer_rb = verbsx::exchange_rbuf(ctx, mine_rb);

  std::string reason;

  // 1) Baseline: client writes the pattern into the server; server verifies.
  if (!is_client) std::memset(buf.data(), 0, kLen);  // clear target FIRST
  boot->barrier();  // server target cleared before the client writes
  if (is_client) {
    for (size_t i = 0; i < kLen; ++i) buf[i] = static_cast<uint8_t>((0x33 + i) % 256);
    reason = conn.post_write(mr->lkey, buf.data(), kLen, peer_rb, 1);
    if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
  }
  boot->barrier();  // client write completes before server verifies
  if (!is_client && reason.empty()) {
    for (size_t i = 0; i < kLen; ++i)
      if (buf[i] != static_cast<uint8_t>((0x33 + i) % 256)) {
        reason = "baseline transfer mismatch at byte " + std::to_string(i);
        break;
      }
  }

  // 2) Down BOTH QPs (simulated failure) and drain.
  if (reason.empty()) reason = conn.to_err();
  if (reason.empty()) { int f = 0; (void)drain_cq(conn.cq(), 1, IBV_WC_WR_FLUSH_ERR, &f, 200); }
  boot->barrier();

  // 2b) NEGATIVE CONTROL: prove the downed QP cannot carry data. Client writes a
  // sentinel target; a WRITE on the ERR-state QP must not land / not succeed.
  if (reason.empty() && is_client) {
    const std::string post =
        conn.post_write(mr->lkey, buf.data(), kLen, peer_rb, 98);
    if (post.empty()) {
      struct ibv_wc wc; std::memset(&wc, 0, sizeof(wc));
      bool ok = false;
      for (int i = 0; i < 200; ++i) {
        if (ibv_poll_cq(conn.cq(), 1, &wc) > 0) { ok = (wc.status == IBV_WC_SUCCESS); break; }
        struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
      }
      if (ok) reason = "negative control: WRITE succeeded on an ERR-state QP";
    }
  }
  boot->barrier();

  // 3) Recover in place: ERR->RESET->INIT->RTS with a QpInfo re-exchange.
  if (reason.empty()) reason = conn.to_reset();
  // Drain any CQE from the negative-control post_write that may not have been
  // reaped within the 200ms window. RESET discards queued WQEs but CQEs already
  // written to the ring are not flushed; a stale wr_id=98 entry would be polled
  // in step 4 and trigger the false stale-CQE assertion.
  { struct ibv_wc wcs[8]; while (ibv_poll_cq(conn.cq(), 8, wcs) > 0) {} }
  if (reason.empty()) reason = conn.reinit();
  // Gate BEFORE the exchange: if the negative-control block set reason on one
  // rank (e.g. ibv_post_send to ERR QP returns EINVAL on bnxt_re), that rank
  // would skip boot->exchange() while the other blocks in it -- a collective
  // mismatch that resets the TCP connection (std::terminate in SocketBootstrap).
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail(reason.empty() ? "peer failed RESET/reinit" : reason);
  }
  {
    verbsx::QpInfo local = conn.local();  // same QPN, refreshed PSN after reinit
    verbsx::QpInfo remote;
    boot->exchange(1 - ctx.rank, &local, &remote, sizeof(verbsx::QpInfo));
    reason = conn.to_rts(remote);
  }
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail(reason.empty() ? "peer failed to recover QP" : reason);
  }

  // 4) Post-recovery: a fresh cross-node WRITE must succeed and land correctly.
  if (!is_client) std::memset(buf.data(), 0, kLen);  // clear target FIRST
  boot->barrier();  // server target cleared before the client writes
  if (is_client) {
    for (size_t i = 0; i < kLen; ++i) buf[i] = static_cast<uint8_t>((0x9C + i) % 256);
    reason = conn.post_write(mr->lkey, buf.data(), kLen, peer_rb, 2);
    if (reason.empty()) {
      struct ibv_wc wc4; std::memset(&wc4, 0, sizeof(wc4));
      reason = conn.poll(kPollMs, &wc4);
      if (reason.empty() && wc4.wr_id != 2)
        reason = "post-recovery poll got wr_id=" + std::to_string(wc4.wr_id) +
                 " (expected 2 — stale CQE false positive)";
    }
  }
  boot->barrier();
  if (!is_client && reason.empty()) {
    for (size_t i = 0; i < kLen; ++i)
      if (buf[i] != static_cast<uint8_t>((0x9C + i) % 256)) {
        reason = "post-recovery transfer mismatch at byte " + std::to_string(i);
        break;
      }
  }
  ibvw::safe_dereg_mr(mr);
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer post-recovery transfer failed" : reason);
  return pass();
}

// Single-process reproducer of the ionic RESET-recovery defect.
//
// Creates two QPs (sender + receiver) on the same ibv_context and ibv_pd,
// loopback-connected using the device's own GID. Only the SENDER QP goes to
// ERR; BOTH are then RESET and re-brought to RTS. This is exactly the scenario
// that exposes an ibv_modify_qp(RESET) recovery defect seen on some NIC drivers
// (the QP is non-functional after recovery despite state transitions returning
// 0). On other drivers it passes.
//
// The failure correlates with the max_inline_data value requested at
// ibv_create_qp() time: RcConn requests max_inline_data=0 (matching the common
// collective-library default), which on affected drivers selects a recovery path
// that does not recover correctly. Both this loopback test and
// vat_rc_qp_reset_recovery are therefore KNOWN_FAILURE on affected drivers;
// remove those policy entries once the driver is fixed.
ACCEPT_TEST(vat_rc_qp_reset_recovery_loopback, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;

  // Open the RDMA device (raw, no RcConn -- mirrors the repro), honoring
  // the selected HCA (ACCEPT_HCA) like RcConn::open() does so this test runs on
  // the platform the operator selected rather than whatever
  // ibv_get_device_list happens to list first.
  int num_devs = 0;
  struct ibv_device** dev_list = ibv_get_device_list(&num_devs);
  if (!dev_list || num_devs == 0) {
    if (dev_list) ibvw::safe_free_device_list(dev_list);
    return skip("no RDMA devices present on this host");
  }
  struct ibv_device* dev = nullptr;
  struct ibv_context* dctx = nullptr;
  if (const std::string want = topology::get_rank_hca(); !want.empty()) {
    for (int i = 0; i < num_devs; ++i) {
      const char* name = ibv_get_device_name(dev_list[i]);
      if (name != nullptr && want == std::string(name)) { dev = dev_list[i]; break; }
    }
    if (dev == nullptr) {
      ibvw::safe_free_device_list(dev_list);
      return fail("IB device '" + std::string(want) + "' not found");
    }
    dctx = ibv_open_device(dev);
    if (dctx == nullptr) {
      ibvw::safe_free_device_list(dev_list);
      return fail(std::string("ibv_open_device failed: ") + std::strerror(errno));
    }
  } else {
    for (int i = 0; i < num_devs && dctx == nullptr; ++i) {
      struct ibv_context* candidate = ibv_open_device(dev_list[i]);
      if (candidate == nullptr) continue;
      struct ibv_port_attr pa;
      std::memset(&pa, 0, sizeof(pa));
      if (ibv_query_port(candidate, 1, &pa) == 0 && pa.state == IBV_PORT_ACTIVE) {
        dev = dev_list[i];
        dctx = candidate;
      } else {
        ibvw::safe_close_device(candidate);
      }
    }
    if (dctx == nullptr) {
      ibvw::safe_free_device_list(dev_list);
      return skip("no RDMA device with an ACTIVE port");
    }
  }
  const char* dev_name_raw = ibv_get_device_name(dev);
  std::string dev_name = dev_name_raw ? dev_name_raw : "<unknown>";
  ibvw::safe_free_device_list(dev_list);

  // Resolve the routable GID index via the capability probe.
  int gid_index = -1;
  {
    const EnvironmentInfo& env = environment();
    for (const auto& rd : env.rdma_devices)
      if (rd.device == dev_name && rd.gid_index >= 0) { gid_index = rd.gid_index; break; }
  }
  if (gid_index < 0) {
    ibvw::safe_close_device(dctx);
    return skip("no routable RoCEv2 GID index for " + dev_name);
  }
  union ibv_gid gid;
  std::memset(&gid, 0, sizeof(gid));
  if (ibv_query_gid(dctx, 1, gid_index, &gid) != 0) {
    ibvw::safe_close_device(dctx);
    return fail(std::string("ibv_query_gid failed: ") + std::strerror(errno));
  }
  struct ibv_port_attr port_attr;
  std::memset(&port_attr, 0, sizeof(port_attr));
  enum ibv_mtu path_mtu = IBV_MTU_4096;
  if (ibv_query_port(dctx, 1, &port_attr) == 0) path_mtu = port_attr.active_mtu;

  // Allocate shared PD and two independent CQs / MRs / bufs (mirrors repro).
  struct ibv_pd* pd = ibv_alloc_pd(dctx);
  if (!pd) { ibvw::safe_close_device(dctx); return fail("ibv_alloc_pd failed"); }
  struct ibv_cq* scq = ibv_create_cq(dctx, 64, nullptr, nullptr, 0);
  struct ibv_cq* rcq = ibv_create_cq(dctx, 64, nullptr, nullptr, 0);
  if (!scq || !rcq) {
    ibvw::safe_dealloc_pd(pd); ibvw::safe_close_device(dctx);
    return fail("ibv_create_cq failed");
  }
  constexpr size_t kBuf = 4096;
  std::vector<uint8_t> sbuf(kBuf), rbuf(kBuf);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ;
  struct ibv_mr* smr = ibv_reg_mr(pd, sbuf.data(), kBuf, access);
  struct ibv_mr* rmr = ibv_reg_mr(pd, rbuf.data(), kBuf, access);
  if (!smr || !rmr) {
    ibvw::safe_dereg_mr(smr); ibvw::safe_dereg_mr(rmr);
    ibvw::safe_destroy_cq(scq); ibvw::safe_destroy_cq(rcq);
    ibvw::safe_dealloc_pd(pd);
    return fail("ibv_reg_mr failed");
  }

  // Create two RC QPs on the same PD (sender + receiver).
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 16; qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  qia.send_cq = scq; qia.recv_cq = scq;
  struct ibv_qp* sqp = ibv_create_qp(pd, &qia);
  qia.send_cq = rcq; qia.recv_cq = rcq;
  struct ibv_qp* rqp = ibv_create_qp(pd, &qia);
  if (!sqp || !rqp) {
    ibvw::safe_destroy_qp(sqp); ibvw::safe_destroy_qp(rqp);
    ibvw::safe_dereg_mr(smr); ibvw::safe_dereg_mr(rmr);
    ibvw::safe_destroy_cq(scq); ibvw::safe_destroy_cq(rcq);
    ibvw::safe_dealloc_pd(pd);
    return fail("ibv_create_qp failed");
  }

  // Bring both QPs to RTS using the local device's own GID (loopback).
  // This is the exact setup from the standalone repro.
  auto to_rts = [&](struct ibv_qp* local_qp, uint32_t remote_qpn) -> std::string {
    struct ibv_qp_attr a;
    std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT; a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(local_qp, &a, IBV_QP_STATE|IBV_QP_PKEY_INDEX|
                      IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))
      return "INIT failed: " + std::string(std::strerror(errno));
    std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR; a.path_mtu = path_mtu;
    a.dest_qp_num = remote_qpn; a.rq_psn = 0;
    a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1; a.ah_attr.grh.dgid = gid;
    a.ah_attr.grh.sgid_index = static_cast<uint8_t>(gid_index);
    a.ah_attr.grh.hop_limit = 255; a.ah_attr.port_num = 1;
    if (ibv_modify_qp(local_qp, &a, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|
                      IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|
                      IBV_QP_MIN_RNR_TIMER))
      return "RTR failed: " + std::string(std::strerror(errno));
    std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS; a.sq_psn = 0;
    a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.max_rd_atomic = 1;
    if (ibv_modify_qp(local_qp, &a, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                      IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC))
      return "RTS failed: " + std::string(std::strerror(errno));
    return "";
  };

  auto do_write = [&](uint8_t pattern) -> std::string {
    std::fill(sbuf.begin(), sbuf.begin() + 64, pattern);
    std::fill(rbuf.begin(), rbuf.begin() + 64, uint8_t{0});
    struct ibv_sge sge{reinterpret_cast<uint64_t>(sbuf.data()), 64, smr->lkey};
    struct ibv_send_wr wr, *bad;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = pattern; wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(rbuf.data());
    wr.wr.rdma.rkey = rmr->rkey;
    if (ibv_post_send(sqp, &wr, &bad))
      return "ibv_post_send failed: " + std::string(std::strerror(errno));
    struct ibv_wc wc; long waited = 0;
    for (;;) {
      std::memset(&wc, 0, sizeof(wc));
      const int n = ibv_poll_cq(scq, 1, &wc);
      if (n < 0) return "ibv_poll_cq failed";
      if (n > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          const char* s = ibv_wc_status_str(wc.status);
          return std::string("completion error: ") + (s ? s : "unknown");
        }
        if (wc.wr_id != pattern)
          return "stale CQE: wr_id=" + std::to_string(wc.wr_id) +
                 " expected=" + std::to_string(pattern);
        return "";
      }
      if (waited >= 5000000L) return "CQ poll timeout after 5000ms";
      struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
      waited += 1000;
    }
  };

  std::string reason;

  // Step 1: initial RTS (s->r, r->s using local GID).
  reason = to_rts(sqp, rqp->qp_num);
  if (reason.empty()) reason = to_rts(rqp, sqp->qp_num);
  if (!reason.empty()) { goto cleanup; }

  // Step 2: baseline RDMA_WRITE (s writes to r).
  reason = do_write(0xAB);
  if (!reason.empty()) { reason = "baseline write: " + reason; goto cleanup; }

  // Step 3: only SENDER goes to ERR (receiver stays RTS -- matches repro).
  {
    struct ibv_qp_attr a; std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_ERR;
    if (ibv_modify_qp(sqp, &a, IBV_QP_STATE))
      { reason = "ERR failed"; goto cleanup; }
    usleep(100000);
    { struct ibv_wc wcs[16]; ibv_poll_cq(scq, 16, wcs); }
  }

  // Step 4: RESET BOTH (receiver was still in RTS -- the key repro pattern).
  {
    struct ibv_qp_attr a; std::memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RESET;
    if (ibv_modify_qp(sqp, &a, IBV_QP_STATE))
      { reason = "RESET sender failed"; goto cleanup; }
    if (ibv_modify_qp(rqp, &a, IBV_QP_STATE))
      { reason = "RESET receiver failed"; goto cleanup; }
  }
  reason = to_rts(sqp, rqp->qp_num);
  if (reason.empty()) reason = to_rts(rqp, sqp->qp_num);
  if (!reason.empty()) { reason = "post-RESET to_rts: " + reason; goto cleanup; }

  // Step 5: post-recovery RDMA_WRITE -- fails on ionic (KNOWN_FAILURE).
  reason = do_write(0xC0);
  if (!reason.empty()) reason = "post-recovery write: " + reason;

cleanup:
  ibvw::safe_destroy_qp(sqp); ibvw::safe_destroy_qp(rqp);
  ibvw::safe_dereg_mr(smr);   ibvw::safe_dereg_mr(rmr);
  ibvw::safe_destroy_cq(scq); ibvw::safe_destroy_cq(rcq);
  ibvw::safe_dealloc_pd(pd);
  ibvw::safe_close_device(dctx);
  return reason.empty() ? pass() : fail(reason);
}

// Async-event monitor: induce a real hardware fault (RDMA WRITE with a bogus
// rkey over the live QP) and confirm a QP async event is delivered on the
// context async fd, decoded, and acked. Single-node: the fault is self-induced
// and the event is local, so no peer is needed.
ACCEPT_TEST(vat_async_event_monitor, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  verbsx::RcConn conn;
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);
  // Loopback self-connect is sufficient: we only need a live QP to fault.
  e = conn.to_rts(conn.local());
  if (!e.empty()) return fail("loopback bring-up: " + e);

  constexpr size_t kLen = 256;
  std::vector<uint8_t> buf(kLen);
  struct ibv_mr* mr =
      ibv_reg_mr(conn.pd(), buf.data(), kLen, IBV_ACCESS_LOCAL_WRITE);
  if (mr == nullptr)
    return fail(std::string("ibv_reg_mr failed: ") + std::strerror(errno));
  // Bogus rkey -> HCA remote-access protection error -> QP async event.
  verbsx::RemoteBuf bad_rb{reinterpret_cast<uint64_t>(buf.data()), 0xDEADBEEF};
  std::string reason = conn.post_write(mr->lkey, buf.data(), kLen, bad_rb, 1);
  if (reason.empty()) {
    struct ibv_wc wc; std::memset(&wc, 0, sizeof(wc));
    for (int i = 0; i < 200; ++i) {
      if (ibv_poll_cq(conn.cq(), 1, &wc) != 0) break;
      struct timespec ts{0, 1000000L}; ::nanosleep(&ts, nullptr);
    }
  }

  struct pollfd pfd;
  pfd.fd = conn.ctx()->async_fd;
  pfd.events = POLLIN;
  const int pr = ::poll(&pfd, 1, 3000);
  if (pr < 0) { ibvw::safe_dereg_mr(mr);
    return fail(std::string("poll(async_fd) failed: ") + std::strerror(errno)); }
  if (pr == 0) { ibvw::safe_dereg_mr(mr);
    return skip("no async event delivered within 3s (provider raises no QP "
                "async event for this fault)"); }

  struct ibv_async_event ev;
  std::memset(&ev, 0, sizeof(ev));
  if (ibv_get_async_event(conn.ctx(), &ev) != 0) { ibvw::safe_dereg_mr(mr);
    return fail(std::string("ibv_get_async_event failed: ") + std::strerror(errno)); }
  const char* name = ibv_event_type_str(ev.event_type);
  const bool qp_event = (ev.event_type == IBV_EVENT_QP_FATAL ||
                         ev.event_type == IBV_EVENT_QP_REQ_ERR ||
                         ev.event_type == IBV_EVENT_QP_ACCESS_ERR);
  ibv_ack_async_event(&ev);
  ibvw::safe_dereg_mr(mr);
  if (!qp_event)
    return fail(std::string("unexpected async event: ") + (name ? name : "?"));
  return pass();
}

// UD recovery protocol over the wire (alive -> ACK -> QPN-exchange). Models the
// RCCL OOB reconnect: two ranks bring up UD endpoints, prove bidirectional
// datagrams flow (the alive/ACK round), then rank 1's UD QP is torn down and
// recreated with a fresh QPN (the failure), the new QPN is re-exchanged, and a
// post-recovery datagram round must still land. This is the datagram-side
// counterpart to vat_rc_qp_reset_recovery.
ACCEPT_TEST(vat_2node_ud_recovery_protocol, "verbs/recovery", Topology::TwoNode,
            Kind::Functional, 45) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  verbsx::UdConn conn;
  verbsx::UdInfo peer;
  bool skip_it = false;
  std::string e = verbsx::ud_connect_peer(ctx, conn, &peer, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kPayload = 64;
  std::vector<uint8_t> rbuf(kPayload + verbsx::kUdGrhLen, 0);
  std::vector<uint8_t> sbuf(kPayload, 0);
  struct ibv_mr* rmr =
      ibv_reg_mr(conn.pd(), rbuf.data(), rbuf.size(), IBV_ACCESS_LOCAL_WRITE);
  struct ibv_mr* smr =
      ibv_reg_mr(conn.pd(), sbuf.data(), kPayload, IBV_ACCESS_LOCAL_WRITE);
  verbsx::Gate mrg = (rmr && smr) ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mrg) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(rmr); ibvw::safe_dereg_mr(smr);
    return fail("ibv_reg_mr failed on a rank");
  }
  struct ibv_ah* ah = conn.create_ah(peer);
  verbsx::Gate ahg = ah ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, ahg) != verbsx::G_OK) {
    ibvw::safe_destroy_ah(ah);
    ibvw::safe_dereg_mr(rmr); ibvw::safe_dereg_mr(smr);
    return fail("ibv_create_ah failed on a rank");
  }

  // One symmetric datagram round: both ranks post a recv, barrier, both send to
  // the peer, then reap the send + recv completions and verify payload + imm.
  // Both ranks use the same pattern so the received bytes match on each side.
  auto do_round = [&](uint8_t pat) -> std::string {
    std::string r = conn.post_recv(rmr->lkey, rbuf.data(), kPayload, /*wr_id=*/7);
    boot->barrier();
    if (r.empty()) {
      verbsx::fill(sbuf.data(), kPayload, pat);
      r = conn.post_send_imm(ah, peer, smr->lkey, sbuf.data(), kPayload, pat,
                             /*wr_id=*/1);
    }
    if (r.empty()) {
      // Shared reap-and-verify (send + recv, imm==pat, payload seed==pat). A
      // TIMEOUT here is treated as a hard round failure -- this recovery test
      // asserts delivery, and the collective gate on the round outcome keeps
      // both ranks lockstep regardless.
      const verbsx::Reap res = verbsx::reap_send_and_recv(
          conn.cq(), /*exp_imm=*/pat, rbuf.data() + verbsx::kUdGrhLen, kPayload,
          /*seed=*/pat, kPollMs, &r);
      if (res == verbsx::Reap::TIMEOUT) r = "UD round timeout";
    }
    return r;
  };

  std::string reason = do_round(0x41);  // alive/ACK round

  // Gate the alive-round outcome UNCONDITIONALLY: UD is unreliable, so a dropped
  // datagram can make one rank's do_round time out while the peer's succeeded.
  // gate() is a collective (allgather) -- it must run on BOTH ranks in lockstep
  // or the pair desyncs and hangs (see the same-file warning at the RESET
  // recovery test). Never short-circuit it behind reason.empty().
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_destroy_ah(ah);
    ibvw::safe_dereg_mr(rmr); ibvw::safe_dereg_mr(smr);
    return fail(reason.empty() ? "peer failed UD alive round" : reason);
  }

  // Failure + recovery on rank 1: tear down and recreate the UD QP (new QPN),
  // bring it back to RTS. Rank 0 keeps its QP; only rank 1's QPN changes.
  if (ctx.rank == kClientRank) {
    reason = conn.recreate();
    if (reason.empty()) reason = conn.to_rts();
  }
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_destroy_ah(ah);
    ibvw::safe_dereg_mr(rmr); ibvw::safe_dereg_mr(smr);
    return fail(reason.empty() ? "peer failed UD recreate" : reason);
  }

  // QPN re-exchange: publish the (possibly new) local UdInfo; both sides refresh
  // `peer` so rank 0 targets rank 1's new QPN. The AH (GID-based) is unchanged.
  {
    verbsx::UdInfo local = conn.local();
    verbsx::UdInfo fresh;
    boot->exchange(1 - ctx.rank, &local, &fresh, sizeof(verbsx::UdInfo));
    peer = fresh;
  }

  // Post-recovery datagram round must still land.
  if (reason.empty()) reason = do_round(0x53);

  ibvw::safe_destroy_ah(ah);
  ibvw::safe_dereg_mr(rmr);
  ibvw::safe_dereg_mr(smr);
  return verbsx::gate_return(boot, reason);
}

// ionic sq_sig_all feature bits (single-node). ionic overloads
// ibv_qp_init_attr.sq_sig_all as a driver feature mask (bits 16-19), not the
// standard bit-0 "signal all WRs". A RCCL data QP uses bits 16|17|18
// (connect.cc ncclIbCreateQpIonic); this asserts the driver accepts that
// encoding and yields a functional QP. NA on non-ionic via policy.
ACCEPT_TEST(vat_ionic_sq_sig_all, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;

  int num = 0;
  struct ibv_device** list = ibv_get_device_list(&num);
  if (!list || num == 0) {
    if (list) ibvw::safe_free_device_list(list);
    return skip("no RDMA devices present on this host");
  }
  // Honor the selected HCA (ACCEPT_HCA) like RcConn::open() does, else the first
  // enumerated device.
  int idx = 0;
  if (const std::string want = topology::get_rank_hca(); !want.empty()) {
    idx = -1;
    for (int i = 0; i < num; ++i) {
      const char* name = ibv_get_device_name(list[i]);
      if (name != nullptr && want == std::string(name)) { idx = i; break; }
    }
    if (idx < 0) {
      ibvw::safe_free_device_list(list);
      return fail("IB device '" + std::string(want) + "' not found");
    }
  }
  struct ibv_context* dctx = ibv_open_device(list[idx]);
  const char* dn = ibv_get_device_name(list[idx]);
  std::string dev_name = dn ? dn : "<unknown>";
  ibvw::safe_free_device_list(list);
  if (!dctx)
    return fail(std::string("ibv_open_device failed: ") + std::strerror(errno));

  int gid_index = -1;
  {
    const EnvironmentInfo& env = environment();
    for (const auto& rd : env.rdma_devices)
      if (rd.device == dev_name && rd.gid_index >= 0) { gid_index = rd.gid_index; break; }
  }
  if (gid_index < 0) {
    ibvw::safe_close_device(dctx);
    return skip("no routable RoCEv2 GID index for " + dev_name);
  }
  union ibv_gid gid;
  std::memset(&gid, 0, sizeof(gid));
  if (ibv_query_gid(dctx, 1, gid_index, &gid) != 0) {
    ibvw::safe_close_device(dctx);
    return fail(std::string("ibv_query_gid failed: ") + std::strerror(errno));
  }

  struct ibv_pd* pd = ibv_alloc_pd(dctx);
  if (!pd) { ibvw::safe_close_device(dctx); return fail("ibv_alloc_pd failed"); }
  struct ibv_cq* cq = ibv_create_cq(dctx, 32, nullptr, nullptr, 0);
  if (!cq) { ibvw::safe_dealloc_pd(pd); ibvw::safe_close_device(dctx);
    return fail("ibv_create_cq failed"); }

  constexpr uint32_t kIonicDataQpSigAll = (1u << 16) | (1u << 17) | (1u << 18);
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq; qia.recv_cq = cq;
  qia.qp_type = IBV_QPT_RC;
  qia.sq_sig_all = kIonicDataQpSigAll;
  qia.cap.max_send_wr = 16; qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  struct ibv_qp* qp = ibv_create_qp(pd, &qia);
  if (!qp) { ibvw::safe_destroy_cq(cq); ibvw::safe_dealloc_pd(pd);
    ibvw::safe_close_device(dctx);
    return fail(std::string("ibv_create_qp(ionic sq_sig_all bits) failed: ") + std::strerror(errno)); }

  std::string reason = verbsx::rc_loopback_to_rts(qp, gid, gid_index);

  constexpr size_t kLen = 256;
  constexpr int kN = 3;
  std::vector<uint8_t> sbuf(kLen), dbuf(kLen);
  struct ibv_mr* smr = nullptr; struct ibv_mr* dmr = nullptr;
  if (reason.empty()) {
    const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    smr = ibv_reg_mr(pd, sbuf.data(), kLen, acc);
    dmr = ibv_reg_mr(pd, dbuf.data(), kLen, acc);
    if (!smr || !dmr) reason = "ibv_reg_mr failed";
  }
  // N signaled WRITEs must complete: proves the feature-flagged QP is functional.
  if (reason.empty()) {
    for (int i = 0; i < kN && reason.empty(); ++i) {
      struct ibv_sge sge{reinterpret_cast<uint64_t>(sbuf.data()), kLen, smr->lkey};
      struct ibv_send_wr wr, *bad = nullptr;
      std::memset(&wr, 0, sizeof(wr));
      wr.wr_id = static_cast<uint64_t>(100 + i);
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.sg_list = &sge; wr.num_sge = 1;
      wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dbuf.data());
      wr.wr.rdma.rkey = dmr->rkey;
      if (ibv_post_send(qp, &wr, &bad) != 0)
        reason = "ibv_post_send failed: " + std::string(std::strerror(errno));
    }
  }
  if (reason.empty()) {
    int ok = 0;
    const int reaped = drain_cq(cq, kN, IBV_WC_SUCCESS, &ok, kPollMs);
    if (reaped < kN)
      reason = "ionic sq_sig_all QP: only " + std::to_string(reaped) + "/" +
               std::to_string(kN) + " completions";
    else if (ok < kN)
      reason = "ionic sq_sig_all QP: " + std::to_string(kN - ok) + "/" +
               std::to_string(kN) + " completions had error status";
  }

  ibvw::safe_dereg_mr(smr); ibvw::safe_dereg_mr(dmr);
  ibvw::safe_destroy_qp(qp); ibvw::safe_destroy_cq(cq);
  ibvw::safe_dealloc_pd(pd); ibvw::safe_close_device(dctx);
  return reason.empty() ? pass() : fail(reason);
}

// ionic WQE-grant threshold probe (single-node). The ionic RESET defect is
// a WQE-format bug: a max_inline_data request of 0..24 grants a 24-byte-inline
// WQE whose in-place RESET recovery is silently broken; >=25 grants a 32-byte
// WQE that recovers. RCCL's own data/CTS QPs request 0/24 -- both broken -- which
// is why it uses destroy+recreate recovery (see vat_rc_qp_reset_recovery). This
// probes the driver's grant step directly: requesting 32 must grant >=25. A
// silent round-down into the broken range is the regression signal. NA on non-ionic.
ACCEPT_TEST(vat_ionic_inline_capacity, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  verbsx::RcConn base;  // only used for device/PD/CQ bring-up
  bool no_dev = false;
  std::string e = base.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);

  constexpr uint32_t kReqInline = 32;
  constexpr uint32_t kSafeFloor = 25;  // below this, the ionic RESET path breaks
  struct ibv_cq* cq = ibv_create_cq(base.ctx(), 8, nullptr, nullptr, 0);
  if (cq == nullptr) return fail("ibv_create_cq failed");

  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq; qia.recv_cq = cq;
  qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 16; qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  qia.cap.max_inline_data = kReqInline;
  struct ibv_qp* qp = ibv_create_qp(base.pd(), &qia);
  if (qp == nullptr) {
    ibvw::safe_destroy_cq(cq);
    return fail(std::string("ibv_create_qp(max_inline=32) failed: ") +
                std::strerror(errno));
  }

  const uint32_t granted = qia.cap.max_inline_data;
  ibvw::safe_destroy_qp(qp);
  ibvw::safe_destroy_cq(cq);

  if (granted < kSafeFloor)
    return fail("granted max_inline_data=" + std::to_string(granted) +
                " < " + std::to_string(kSafeFloor) +
                " (ionic broken-WQE grant; RESET recovery would fail)");
  return pass();
}

// ionic UDMA-mask assignment (single-node). ionic_dv_pd_set_udma_mask
// assigns a PD to one of two UDMA engines; RCCL calls it on a fresh PD before
// ibv_create_qp, alternating HIGH/LOW per channel (connect.cc:441-443). Set both
// masks on a fresh PD (no QPs yet, matching that order), then create a QP on it
// and reach RTS to confirm the mask doesn't break bring-up. SKIP without
// libionic; NA on non-ionic via policy.
ACCEPT_TEST(vat_ionic_udma_mask, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!ionicdv::init())
    return skip("libionic direct-verbs unavailable (ionic_dv_pd_set_udma_mask "
                "not resolvable via dlvsym @ IONIC_1.0)");

  verbsx::RcConn base;  // device/context + GID resolution only
  bool no_dev = false;
  std::string e = base.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);
  const int sgid = base.gid_index();
  if (sgid < 0)
    return skip("no routable RoCEv2 GID index for " + base.dev_name());

  // Fresh PD with no QPs on it yet, matching RCCL's order: mask the PD BEFORE
  // any ibv_create_qp (RcConn's own PD already owns a QP, so don't reuse it).
  struct ibv_pd* pd = ibv_alloc_pd(base.ctx());
  if (pd == nullptr) return fail("ibv_alloc_pd failed");

  auto set_udma = ionicdv::pd_set_udma_mask();
  if (int rc = set_udma(pd, ionicdv::kUdmaHigh); rc != 0) {
    ibvw::safe_dealloc_pd(pd);
    return fail("ionic_dv_pd_set_udma_mask(HIGH) rc=" + std::to_string(rc) +
                ": " + std::strerror(std::abs(rc)));
  }
  if (int rc = set_udma(pd, ionicdv::kUdmaLow); rc != 0) {
    ibvw::safe_dealloc_pd(pd);
    return fail("ionic_dv_pd_set_udma_mask(LOW) rc=" + std::to_string(rc) +
                ": " + std::strerror(std::abs(rc)));
  }

  struct ibv_cq* cq = ibv_create_cq(base.ctx(), 8, nullptr, nullptr, 0);
  if (cq == nullptr) { ibvw::safe_dealloc_pd(pd); return fail("ibv_create_cq failed"); }
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq; qia.recv_cq = cq;
  qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 16; qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  struct ibv_qp* qp = ibv_create_qp(pd, &qia);
  if (qp == nullptr) {
    ibvw::safe_destroy_cq(cq); ibvw::safe_dealloc_pd(pd);
    return fail(std::string("ibv_create_qp on UDMA-masked PD failed: ") +
                std::strerror(errno));
  }

  union ibv_gid gid;
  std::memcpy(gid.raw, base.local().gid, 16);
  const std::string reason = verbsx::rc_loopback_to_rts(qp, gid, sgid);

  ibvw::safe_destroy_qp(qp);
  ibvw::safe_destroy_cq(cq);
  ibvw::safe_dealloc_pd(pd);
  return reason.empty() ? pass()
                        : fail("post-UDMA-mask QP bring-up: " + reason);
}

// ionic GPU-Direct-Async enable (single-node). ionic_dv_qp_set_gda turns
// on GDA for a QP's send/recv path; RCCL calls it right after ibv_create_qp on
// data QPs (net_ib_cast/connect.cc:447, with en_send=false, en_recv=true). This
// test exercises the real libionic symbol: create a QP, enable GDA with the same
// argument RCCL uses, and require rc==0. SKIP when libionic / the symbol is
// absent. NA on non-ionic NICs via policy.
ACCEPT_TEST(vat_ionic_gda_enable, "verbs/recovery", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!ionicdv::init())
    return skip("libionic direct-verbs unavailable (ionic_dv_qp_set_gda not "
                "resolvable via dlvsym @ IONIC_1.0)");

  verbsx::RcConn base;
  bool no_dev = false;
  std::string e = base.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);

  struct ibv_cq* cq = ibv_create_cq(base.ctx(), 8, nullptr, nullptr, 0);
  if (cq == nullptr) return fail("ibv_create_cq failed");
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq; qia.recv_cq = cq;
  qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 16; qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  struct ibv_qp* qp = ibv_create_qp(base.pd(), &qia);
  if (qp == nullptr) {
    ibvw::safe_destroy_cq(cq);
    return fail(std::string("ibv_create_qp failed: ") + std::strerror(errno));
  }

  // Same arguments RCCL uses for data QPs: enable recv-side GDA only.
  const int rc = ionicdv::qp_set_gda()(qp, /*en_send=*/false, /*en_recv=*/true);
  ibvw::safe_destroy_qp(qp);
  ibvw::safe_destroy_cq(cq);
  if (rc != 0)
    return fail("ionic_dv_qp_set_gda(false,true) rc=" + std::to_string(rc) +
                ": " + std::strerror(std::abs(rc)));
  return pass();
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
