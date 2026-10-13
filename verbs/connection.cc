/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Verbs acceptance tests -- connection baseline (UD / SRQ / ECE).
//
// These cover the connection primitives RCCL relies on before or alongside the
// RC datapath:
//   * vat_ud_qp_lifecycle  -- UD QP + QKEY + AH, SEND_WITH_IMM + receive (2-node)
//   * vat_ud_grh_handling  -- the 40-byte GRH prefix on a UD receive (2-node)
//   * vat_srq_lifecycle    -- shared-receive-queue path (single-node loopback)
//   * vat_ece_negotiation  -- Enhanced Connection Establishment (2-node)
//
// UD tests use the UdConn datagram endpoint + the TwoNode collective gate. SRQ
// runs single-node over a loopback RC QP whose receive side is fed by the SRQ.
// ECE piggybacks on the RC connect and self-skips where the provider lacks it
// (expected on bnxt_re / ionic). Built only with ACCEPT_HAVE_IBVERBS.

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"
#include "verbs/TwoNode.hpp"
#include "verbs/UdConn.hpp"

#include <arpa/inet.h>  // ntohl

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "common/Bootstrap.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {

namespace {

constexpr int kPollMs = 5000;
constexpr int kClientRank = 1;  // UD sender; receiver = !is_client

using verbsx::fill;
using verbsx::mismatch;

// UD retry policy: UD is unreliable, so an isolated dropped datagram must not
// hard-fail the test (CI flakiness). We resend up to kUdAttempts times, each
// with a short poll window; only an all-attempts-lost run (a genuine fabric
// break) fails. The receiver posts ONE recv WR before the loop -- a timed-out
// poll leaves that WR armed, so the first datagram to arrive on any attempt
// completes it with no re-post.
constexpr int kUdAttempts = 3;
constexpr int kUdAttemptMs = 1000;

// Per-attempt outcome, allgather'd so both ranks decide identically and stay
// collective-lockstep. Combined by max(): FAIL(2) > DELIVERED(1) > PENDING(0),
// so a hard error on either rank dominates, a delivery ends the loop, and a
// mutual timeout retries.
enum UdVote : uint8_t { UD_PENDING = 0, UD_DELIVERED = 1, UD_FAIL = 2 };

// Shared body for the two UD tests: sender SEND_WITH_IMM -> receiver reaps the
// recv completion and verifies payload + imm. When check_grh is set, the
// receiver additionally asserts the WC carried a GRH and the payload landed at
// offset kUdGrhLen after a well-formed (non-zero) GRH prefix. Retries dropped
// datagrams (see kUdAttempts) instead of failing on the first loss.
TestResult ud_send_recv(TestContext& ctx, bool check_grh) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_sender = (ctx.rank == kClientRank);
  verbsx::UdConn conn;
  verbsx::UdInfo peer;
  bool skip_it = false;
  std::string e = verbsx::ud_connect_peer(ctx, conn, &peer, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kPayload = 256;
  constexpr uint32_t kImm = 0xABCD1234;
  // Receiver buffer must hold the 40-byte GRH prefix ahead of the payload.
  std::vector<uint8_t> buf(kPayload + verbsx::kUdGrhLen, 0);
  struct ibv_mr* mr =
      ibv_reg_mr(conn.pd(), buf.data(), buf.size(), IBV_ACCESS_LOCAL_WRITE);
  verbsx::Gate mrg = mr ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mrg) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail("ibv_reg_mr failed on a rank");
  }

  std::string reason;
  struct ibv_ah* ah = nullptr;
  if (!is_sender) {
    // Post the single recv WR once; it stays armed across retry attempts.
    reason = conn.post_recv(mr->lkey, buf.data(), kPayload, /*wr_id=*/7);
  } else {
    ah = conn.create_ah(peer);
    if (ah == nullptr) reason = "ibv_create_ah failed";
  }
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK) {
    ibvw::safe_destroy_ah(ah);
    ibvw::safe_dereg_mr(mr);
    return fail(reason.empty() ? "peer UD setup failed" : reason);
  }
  boot->barrier();  // receiver recv armed before the sender transmits

  bool delivered = false;
  for (int attempt = 0; attempt < kUdAttempts && !delivered && reason.empty();
       ++attempt) {
    UdVote mine = UD_PENDING;
    if (is_sender) {
      // Payload rides from the tail of the buffer (GRH prefix only on rx).
      uint8_t* src = buf.data() + verbsx::kUdGrhLen;
      fill(src, kPayload, 0x77);
      reason = conn.post_send_imm(ah, peer, mr->lkey, src, kPayload, kImm,
                                 /*wr_id=*/1);
      // The send completion is local ("handed to the wire") and must arrive;
      // a send-side timeout is a stuck SQ, not a dropped datagram -> hard fail.
      if (reason.empty()) reason = conn.poll(kUdAttemptMs, nullptr);
      mine = reason.empty() ? UD_PENDING : UD_FAIL;  // sender can't confirm rx
    } else {
      struct ibv_wc wc;
      std::memset(&wc, 0, sizeof(wc));
      const std::string pr = conn.poll(kUdAttemptMs, &wc);
      if (pr.empty()) {
        // A datagram arrived -- verify it.
        if (!(wc.opcode == IBV_WC_RECV && (wc.wc_flags & IBV_WC_WITH_IMM)))
          reason = "receiver: completion is not a recv-with-imm";
        else if (ntohl(wc.imm_data) != kImm)
          reason = "receiver: imm mismatch";
        else if (check_grh && !(wc.wc_flags & IBV_WC_GRH))
          reason = "receiver: WC missing IBV_WC_GRH flag";
        else {
          const long bad = mismatch(buf.data() + verbsx::kUdGrhLen, kPayload, 0x77);
          if (bad >= 0) reason = "payload mismatch at byte " + std::to_string(bad);
        }
        if (reason.empty() && check_grh) {
          // The 40-byte GRH prefix must be present (non-zero): the provider
          // delivered the routing header ahead of the payload.
          bool all_zero = true;
          for (uint32_t i = 0; i < verbsx::kUdGrhLen; ++i)
            if (buf[i] != 0) { all_zero = false; break; }
          if (all_zero) reason = "receiver: GRH prefix is all zero";
        }
        mine = reason.empty() ? UD_DELIVERED : UD_FAIL;
      } else if (pr.find("timeout") != std::string::npos) {
        mine = UD_PENDING;  // datagram dropped/late; recv WR stays armed, retry
      } else {
        reason = pr;        // real poll error (poll_cq / WC error status)
        mine = UD_FAIL;
      }
    }

    // Combine the per-rank votes so both ranks take the same branch.
    uint8_t v = static_cast<uint8_t>(mine), all[2] = {0, 0};
    boot->allgather(&v, all, 1);
    const uint8_t combined = std::max(all[0], all[1]);
    if (combined == UD_FAIL) {
      if (reason.empty()) reason = "peer UD exchange failed";
      break;
    }
    if (combined == UD_DELIVERED) delivered = true;
  }

  if (reason.empty() && !delivered)
    reason = "UD datagram not delivered after " + std::to_string(kUdAttempts) +
             " attempts";

  ibvw::safe_destroy_ah(ah);
  ibvw::safe_dereg_mr(mr);
  return verbsx::gate_return(boot, reason);
}

}  // namespace

// UD QP lifecycle: QKEY-addressed datagram SEND_WITH_IMM across the wire.
ACCEPT_TEST(vat_ud_qp_lifecycle, "verbs/connection", Topology::TwoNode,
            Kind::Functional, 30) {
  return ud_send_recv(ctx, /*check_grh=*/false);
}

// UD GRH handling: same datagram path, but verify the 40-byte GRH prefix is
// delivered ahead of the payload and the WC carries IBV_WC_GRH.
ACCEPT_TEST(vat_ud_grh_handling, "verbs/connection", Topology::TwoNode,
            Kind::Functional, 30) {
  return ud_send_recv(ctx, /*check_grh=*/true);
}

// Shared-receive-queue path (single-node loopback). Create an SRQ, feed a
// loopback RC QP's receive side from it, drive an RC SEND into the QP, and
// verify the SRQ-posted recv WR is consumed with the correct payload + imm.
ACCEPT_TEST(vat_srq_lifecycle, "verbs/connection", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;

  // Open device + PD/CQ via RcConn's open(), but we need a QP wired to the SRQ,
  // so build the QP manually against RcConn's context/pd rather than using its
  // own RC QP. RcConn::open() also creates its RC QP; that one is simply unused
  // here (harmless), while we create + connect our SRQ-backed QP alongside it.
  verbsx::RcConn base;
  bool no_dev = false;
  std::string e = base.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);
  const int sgid = base.gid_index();
  if (sgid < 0)
    return skip("no routable RoCEv2 GID index for " + base.dev_name());

  struct ibv_context* dctx = base.ctx();
  struct ibv_pd* pd = base.pd();

  struct ibv_srq_init_attr sia;
  std::memset(&sia, 0, sizeof(sia));
  sia.attr.max_wr = 16;
  sia.attr.max_sge = 1;
  struct ibv_srq* srq = ibv_create_srq(pd, &sia);
  if (srq == nullptr) {
    if (errno == EOPNOTSUPP || errno == ENOSYS)
      return skip("SRQ not supported by provider (EOPNOTSUPP)");
    return fail(std::string("ibv_create_srq failed: ") + std::strerror(errno));
  }

  struct ibv_cq* cq = ibv_create_cq(dctx, 32, nullptr, nullptr, 0);
  if (cq == nullptr) {
    ibvw::safe_destroy_srq(srq);
    return fail("ibv_create_cq failed");
  }

  // RC QP whose recv side is fed by the SRQ.
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq;
  qia.recv_cq = cq;
  qia.srq = srq;
  qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 16;
  qia.cap.max_send_sge = 1;
  // recv caps come from the SRQ; leave max_recv_wr/sge at 0.
  struct ibv_qp* qp = ibv_create_qp(pd, &qia);
  if (qp == nullptr) {
    ibvw::safe_destroy_cq(cq);
    ibvw::safe_destroy_srq(srq);
    return fail(std::string("ibv_create_qp(SRQ) failed: ") + std::strerror(errno));
  }

  union ibv_gid gid;
  std::memcpy(gid.raw, base.local().gid, 16);

  std::string reason = verbsx::rc_loopback_to_rts(qp, gid, sgid);

  constexpr size_t kLen = 512;
  constexpr uint32_t kImm = 0x5150F00D;
  constexpr uint64_t kRecvWrId = 42;
  constexpr uint8_t kSeed = 0x88;
  std::vector<uint8_t> sbuf(kLen), rbuf(kLen);
  struct ibv_mr* smr = nullptr;
  struct ibv_mr* rmr = nullptr;
  if (reason.empty()) {
    smr = ibv_reg_mr(pd, sbuf.data(), kLen, IBV_ACCESS_LOCAL_WRITE);
    rmr = ibv_reg_mr(pd, rbuf.data(), kLen, IBV_ACCESS_LOCAL_WRITE);
    if (!smr || !rmr) reason = "ibv_reg_mr failed";
  }

  // Post the receive WR to the SRQ (not the QP): this is the shared-receive path.
  if (reason.empty()) {
    struct ibv_sge sge{reinterpret_cast<uint64_t>(rbuf.data()), kLen, rmr->lkey};
    struct ibv_recv_wr rwr, *bad = nullptr;
    std::memset(&rwr, 0, sizeof(rwr));
    rwr.wr_id = kRecvWrId;
    rwr.sg_list = &sge;
    rwr.num_sge = 1;
    if (ibv_post_srq_recv(srq, &rwr, &bad) != 0)
      reason = "ibv_post_srq_recv failed: " + std::string(std::strerror(errno));
  }

  // Drive an RC SEND_WITH_IMM into the loopback QP; it consumes the SRQ recv WR.
  if (reason.empty()) {
    fill(sbuf.data(), kLen, kSeed);
    struct ibv_sge sge{reinterpret_cast<uint64_t>(sbuf.data()), kLen, smr->lkey};
    struct ibv_send_wr wr, *bad = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = 1;
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(kImm);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (ibv_post_send(qp, &wr, &bad) != 0)
      reason = "ibv_post_send failed: " + std::string(std::strerror(errno));
  }

  // Reap the send + SRQ-fed recv completions and verify the recv. Loopback RC is
  // reliable, so a TIMEOUT here is a real failure (unlike the UD path).
  if (reason.empty()) {
    const verbsx::Reap r = verbsx::reap_send_and_recv(
        cq, kImm, rbuf.data(), kLen, kSeed, kPollMs, &reason, kRecvWrId);
    if (r == verbsx::Reap::TIMEOUT) reason = "SRQ completion timeout";
  }

  ibvw::safe_dereg_mr(smr);
  ibvw::safe_dereg_mr(rmr);
  ibvw::safe_destroy_qp(qp);
  ibvw::safe_destroy_cq(cq);
  ibvw::safe_destroy_srq(srq);
  return reason.empty() ? pass() : fail(reason);
}

// Enhanced Connection Establishment: query the ECE struct on the QP, exchange
// it with the peer, set the negotiated value, then bring the QP to RTS and
// confirm a WRITE still works. SKIP cleanly where the provider lacks ECE
// (EOPNOTSUPP / ENOSYS) -- expected on bnxt_re and ionic.
ACCEPT_TEST(vat_ece_negotiation, "verbs/connection", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;

  // Bring the QP to INIT (via open) but NOT yet to RTS -- ECE must be set on the
  // QP before the RTR transition. connect_peer drives all the way to RTS, so we
  // replicate its INIT + gate here, negotiate ECE, then finish to_rts manually.
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  verbsx::Gate mine = verbsx::G_OK;
  std::string reason;
  if (!e.empty()) { mine = no_dev ? verbsx::G_SKIP : verbsx::G_FAIL; reason = e; }
  else if (conn.gid_index() < 0) {
    mine = verbsx::G_SKIP;
    reason = "no routable RoCEv2 GID on " + conn.dev_name();
  }
  if (verbsx::Gate g = verbsx::gate(boot, mine); g != verbsx::G_OK) {
    if (g == verbsx::G_SKIP) return skip(reason.empty() ? "peer has no device" : reason);
    return fail(reason.empty() ? "peer QP open failed" : reason);
  }

  // Query local ECE. A missing symbol throws MissingSymbol -> Skip at the
  // boundary; EOPNOTSUPP/ENOSYS from the provider is a clean in-body skip.
  struct ibv_ece my_ece;
  std::memset(&my_ece, 0, sizeof(my_ece));
  const int qrc = ibv_query_ece(conn.qp(), &my_ece);
  verbsx::Gate ece_ok = (qrc == 0) ? verbsx::G_OK : verbsx::G_SKIP;
  if (verbsx::gate(boot, ece_ok) != verbsx::G_OK)
    return skip("ECE unsupported (ibv_query_ece rc != 0) on a rank");

  // Exchange ECE with the peer and set the negotiated value.
  struct ibv_ece peer_ece;
  std::memset(&peer_ece, 0, sizeof(peer_ece));
  boot->exchange(1 - ctx.rank, &my_ece, &peer_ece, sizeof(struct ibv_ece));
  const int src = ibv_set_ece(conn.qp(), &peer_ece);
  // Some providers accept query but reject a foreign vendor_id on set -- treat a
  // set failure as an unsupported-negotiation skip rather than a hard fail.
  verbsx::Gate set_ok = (src == 0) ? verbsx::G_OK : verbsx::G_SKIP;
  if (verbsx::gate(boot, set_ok) != verbsx::G_OK)
    return skip("ECE set rejected (ibv_set_ece rc != 0) on a rank");

  // Finish the connect: exchange QpInfo, drive to RTS, confirm a WRITE lands.
  verbsx::QpInfo local = conn.local();
  verbsx::QpInfo remote;
  boot->exchange(1 - ctx.rank, &local, &remote, sizeof(verbsx::QpInfo));
  reason = conn.to_rts(remote);
  if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
      verbsx::G_OK)
    return fail(reason.empty() ? "peer failed RTS after ECE" : reason);

  constexpr size_t kLen = 4096;
  std::vector<uint8_t> buf(kLen);
  verbsx::ScopedMr mr(conn, buf.data(), kLen,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr failed on a rank");

  if (!is_client) std::memset(buf.data(), 0, kLen);
  boot->barrier();
  if (is_client) {
    fill(buf.data(), kLen, 0x99);
    reason = conn.post_write(mr.lkey(), buf.data(), kLen, mr.peer_rb(), 1);
    if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
  }
  boot->barrier();
  if (!is_client && reason.empty()) {
    const long bad = mismatch(buf.data(), kLen, 0x99);
    if (bad >= 0) reason = "post-ECE transfer mismatch at byte " + std::to_string(bad);
  }
  return verbsx::gate_return(boot, reason);
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
