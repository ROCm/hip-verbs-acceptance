/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Verbs acceptance tests -- host-memory data path (Group A), REAL 2-node.
//
// RCCL's datapath is node-to-node RDMA, so these tests connect two ranks over
// common/Bootstrap and drive RDMA across the wire (rank 1 = client/initiator,
// rank 0 = server/target) rather than a loopback self-connect. Each is
// Topology::TwoNode and skips cleanly when run with a single rank.
//
// Connection bring-up + the collective gate are factored into verbs/TwoNode.hpp
// (shared with the recovery suite). ScopedMr handles MR registration, the
// collective gate check, and RemoteBuf exchange in one step. Built only with
// ACCEPT_HAVE_IBVERBS.

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"
#include "verbs/TwoNode.hpp"
#include "verbs/VerbsPerf.hpp"

#include <arpa/inet.h>  // ntohl

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "common/Bootstrap.hpp"
#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {

namespace {

constexpr int kPollMs = 5000;  // generous: covers chained 8×4K on a slow fabric
constexpr int kClientRank = 1;  // initiator; server (target) = !is_client

using verbsx::fill;
using verbsx::mismatch;

// gate_return + fold this rank's perf into the result on a unanimous Pass only
// (fail/skip keep their reason). perf is empty off --kind perf|all.
inline TestResult gate_return_perf(Bootstrap* boot, const std::string& reason,
                                   PerfSink perf) {
  TestResult result = verbsx::gate_return(boot, reason);
  if (result.status == Status::Pass) result.perf = std::move(perf);
  return result;
}

}  // namespace

// RC QP INIT->RTR->RTS across the wire -- the reliable-connection baseline.
ACCEPT_TEST(vat_rc_qp_lifecycle, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);
  return pass();
}

// RDMA_WRITE across the wire with a data-integrity check over a size sweep.
ACCEPT_TEST(vat_rdma_write, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kMax = 256 * 1024;
  std::vector<uint8_t> buf(kMax);
  verbsx::ScopedMr mr(conn, buf.data(), kMax,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr failed on a rank");

  std::string reason;
  PerfSink perf;
  // 64B is included for functional coverage; its GB/s reads near-zero (expected
  // for small messages -- the per-op latency is the meaningful metric there).
  for (size_t len : {size_t{64}, size_t{4096}, size_t{65536}, kMax}) {
    if (!is_client) std::memset(buf.data(), 0, len);
    boot->barrier();
    if (is_client) {
      fill(buf.data(), len, 0x11);
      // PerfLoop: 1 pass functional, warmup+N in perf mode -> per-size bandwidth.
      // Server verifies the final buffer post-barrier, so repeated writes are safe.
      PerfLoop loop;
      for (auto pass : loop) {
        (void)pass;
        reason = conn.post_write(mr.lkey(), buf.data(),
                                 static_cast<uint32_t>(len), mr.peer_rb(), 1);
        if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
        if (!reason.empty()) break;
      }
      if (reason.empty()) loop.emit_bandwidth(perf, "rdma_write.bw", len);
    }
    boot->barrier();
    if (!is_client && reason.empty()) {
      const long bad = mismatch(buf.data(), len, 0x11);
      if (bad >= 0)
        reason = "len " + std::to_string(len) + " mismatch at byte " +
                 std::to_string(bad);
    }
    if (verbsx::gate(boot, reason.empty() ? verbsx::G_OK : verbsx::G_FAIL) !=
        verbsx::G_OK) {
      if (reason.empty()) reason = "peer failed at len " + std::to_string(len);
      break;
    }
  }
  return gate_return_perf(boot, reason, std::move(perf));
}

// RDMA_READ across the wire: client reads the server's pre-filled window.
ACCEPT_TEST(vat_rdma_read, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 8192;
  std::vector<uint8_t> buf(kLen);
  if (is_client)
    std::memset(buf.data(), 0, kLen);
  else
    fill(buf.data(), kLen, 0x22);
  verbsx::ScopedMr mr(conn, buf.data(), kLen,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr failed on a rank");

  boot->barrier();  // server's buffer is filled before the client reads
  std::string reason;
  PerfSink perf;
  if (is_client) {
    // Server buffer is static, so perf_rdma's warmup+N repeated reads are safe.
    reason = perf_rdma(perf, "rdma_read.lat", conn, RdmaOp::Read, mr.lkey(),
                       buf.data(), kLen, mr.peer_rb(), kPollMs);
    if (reason.empty()) {
      const long bad = mismatch(buf.data(), kLen, 0x22);
      if (bad >= 0) reason = "read data mismatch at byte " + std::to_string(bad);
    }
  }
  boot->barrier();
  return gate_return_perf(boot, reason, std::move(perf));
}

// RDMA_WRITE_WITH_IMM: client writes+imm into the server's window; server reaps
// the imm on a recv completion and verifies the payload.
ACCEPT_TEST(vat_rdma_write_with_imm, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 4096;
  constexpr uint32_t kImm = 0xCAFEF00D;
  std::vector<uint8_t> buf(kLen);
  verbsx::ScopedMr mr(conn, buf.data(), kLen,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr failed on a rank");

  std::string reason;
  if (!is_client) {
    std::memset(buf.data(), 0, kLen);
    reason = conn.post_recv(mr.lkey(), buf.data(), kLen, /*wr_id=*/2);
  } else {
    fill(buf.data(), kLen, 0x33);
  }
  boot->barrier();  // server recv armed before the client write-with-imm

  if (is_client && reason.empty()) {
    reason = conn.post_write_imm(mr.lkey(), buf.data(), kLen,
                                 mr.peer_rb(), kImm, 1);
    if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
  } else if (!is_client && reason.empty()) {
    struct ibv_wc wc;
    std::memset(&wc, 0, sizeof(wc));
    reason = conn.poll(kPollMs, &wc);
    if (reason.empty()) {
      if (!(wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
            (wc.wc_flags & IBV_WC_WITH_IMM)))
        reason = "server: completion is not a recv-with-imm";
      else if (ntohl(wc.imm_data) != kImm)
        reason = "server: imm mismatch";
      else {
        const long bad = mismatch(buf.data(), kLen, 0x33);
        if (bad >= 0) reason = "payload mismatch at byte " + std::to_string(bad);
      }
    }
  }
  return verbsx::gate_return(boot, reason);
}

// Chained RDMA_WRITEs (linked WR list, only the last signaled) across the wire.
ACCEPT_TEST(vat_rdma_write_chain, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr int kN = 8;
  constexpr size_t kSeg = 4096;
  const size_t total = kSeg * kN;
  std::vector<uint8_t> buf(total);
  verbsx::ScopedMr mr(conn, buf.data(), total,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr failed on a rank");

  std::string reason;
  if (!is_client) std::memset(buf.data(), 0, total);
  boot->barrier();
  if (is_client) {
    fill(buf.data(), total, 0x44);
    struct ibv_sge sge[kN];
    struct ibv_send_wr wr[kN];
    std::memset(sge, 0, sizeof(sge));
    std::memset(wr, 0, sizeof(wr));
    for (int i = 0; i < kN; ++i) {
      const size_t off = static_cast<size_t>(i) * kSeg;
      sge[i].addr = reinterpret_cast<uint64_t>(buf.data() + off);
      sge[i].length = kSeg;
      sge[i].lkey = mr.lkey();
      wr[i].wr_id = static_cast<uint64_t>(100 + i);
      wr[i].opcode = IBV_WR_RDMA_WRITE;
      wr[i].sg_list = &sge[i];
      wr[i].num_sge = 1;
      wr[i].wr.rdma.remote_addr = mr.peer_rb().addr + off;
      wr[i].wr.rdma.rkey = mr.peer_rb().rkey;
      wr[i].next = (i + 1 < kN) ? &wr[i + 1] : nullptr;
      wr[i].send_flags = (i + 1 == kN) ? IBV_SEND_SIGNALED : 0;
    }
    reason = conn.post_chain(&wr[0]);
    if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
  }
  boot->barrier();
  if (!is_client && reason.empty()) {
    const long bad = mismatch(buf.data(), total, 0x44);
    if (bad >= 0) reason = "chain mismatch at byte " + std::to_string(bad);
  }
  return verbsx::gate_return(boot, reason);
}

// Small RDMA_WRITE with IBV_SEND_INLINE across the wire (no source MR/lkey).
ACCEPT_TEST(vat_inline_send, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  constexpr size_t kLen = 24;
  std::vector<uint8_t> buf(kLen);
  verbsx::ScopedMr mr(conn, buf.data(), kLen,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr failed on a rank");

  std::string reason;
  PerfSink perf;
  uint8_t src[kLen];
  if (!is_client) std::memset(buf.data(), 0, kLen);
  boot->barrier();
  if (is_client) {
    fill(src, kLen, 0x55);
    // Inline WRITE has no perf_rdma variant (no lkey); drive PerfLoop directly.
    PerfLoop loop;
    for (auto pass : loop) {
      (void)pass;
      reason = conn.post_write_inline(src, kLen, mr.peer_rb(), /*wr_id=*/1);
      if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
      if (!reason.empty()) break;
    }
    if (reason.empty()) loop.emit_latency(perf, "inline_send.lat");
  }
  boot->barrier();
  if (!is_client && reason.empty()) {
    const long bad = mismatch(buf.data(), kLen, 0x55);
    if (bad >= 0) reason = "inline mismatch at byte " + std::to_string(bad);
  }
  return gate_return_perf(boot, reason, std::move(perf));
}

// RDMA ATOMIC_FETCH_AND_ADD across the wire. Asymmetric MRs (server: counter
// with REMOTE_ATOMIC; client: fetch slot with LOCAL_WRITE only) so ScopedMr
// doesn't apply -- register manually.
ACCEPT_TEST(vat_atomic_fetch_add, "verbs/datapath", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_client = (ctx.rank == kClientRank);
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string e = verbsx::connect_peer(ctx, conn, &skip_it);
  if (!e.empty()) return skip_it ? skip(e) : fail(e);

  alignas(8) uint64_t counter = 0x00FF;   // server: atomic target
  alignas(8) uint64_t fetched = 0;        // client: fetched pre-add value
  const uint64_t add = 0x0002;
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  struct ibv_mr* mr =
      is_client
          ? ibv_reg_mr(conn.pd(), &fetched, sizeof(fetched), IBV_ACCESS_LOCAL_WRITE)
          : ibv_reg_mr(conn.pd(), &counter, sizeof(counter), access);
  verbsx::Gate mine = mr ? verbsx::G_OK : verbsx::G_FAIL;
  if (verbsx::gate(boot, mine) != verbsx::G_OK) {
    ibvw::safe_dereg_mr(mr);
    return fail("ibv_reg_mr failed on a rank");
  }

  // Server publishes the counter rkey; client passes rkey=0 (it's the initiator,
  // not the target -- the server never needs to RDMA into the client's fetch slot).
  verbsx::RemoteBuf mine_rb{reinterpret_cast<uint64_t>(&counter),
                            is_client ? 0u : mr->rkey};
  verbsx::RemoteBuf peer_rb = verbsx::exchange_rbuf(ctx, mine_rb);

  boot->barrier();
  std::string reason;
  if (is_client) {
    reason = conn.post_atomic_fadd(mr->lkey, &fetched, peer_rb, add, /*wr_id=*/1);
    if (reason.empty()) reason = conn.poll(kPollMs, nullptr);
    if (reason.empty() && fetched != 0x00FF)
      reason = "fetched pre-add " + std::to_string(fetched) + " != 255";
  }
  boot->barrier();
  if (!is_client && reason.empty() && counter != 0x0101)
    reason = "post-add counter " + std::to_string(counter) + " != 257";
  ibvw::safe_dereg_mr(mr);
  return verbsx::gate_return(boot, reason);
}

// Register an MR via the explicit-IOVA ibv_reg_mr_iova2 path and RDMA WRITE
// over it. EFAULT on a valid buffer is the regression signal.
// Note: the ibv_reg_mr macro in our wrapper redirects to ibv_reg_mr_iova2
// (with iova == addr), so ScopedMr exercises that path implicitly.
ACCEPT_TEST(vat_reg_mr_iova2, "verbs/datapath", Topology::TwoNode,
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
  verbsx::ScopedMr mr(conn, buf.data(), kLen,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
                      boot, ctx);
  if (!mr) return fail("ibv_reg_mr_iova2 failed on a rank");

  std::string reason;
  PerfSink perf;
  if (!is_client) std::memset(buf.data(), 0, kLen);
  boot->barrier();
  if (is_client) {
    fill(buf.data(), kLen, 0x66);
    reason = perf_rdma(perf, "rdma_write_iova2.lat", conn, RdmaOp::Write,
                       mr.lkey(), buf.data(), kLen, mr.peer_rb(), kPollMs);
  }
  boot->barrier();
  if (!is_client && reason.empty()) {
    const long bad = mismatch(buf.data(), kLen, 0x66);
    if (bad >= 0) reason = "iova2 transfer mismatch at byte " + std::to_string(bad);
  }
  return gate_return_perf(boot, reason, std::move(perf));
}

// Address-handle create/destroy in isolation -- single-node, no peer traffic.
ACCEPT_TEST(vat_ah_lifecycle, "verbs/datapath", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  verbsx::RcConn conn;
  bool no_dev = false;
  std::string e = conn.open("", no_dev);
  if (!e.empty()) return no_dev ? skip(e) : fail(e);

  const int sgid_index = conn.gid_index();
  if (sgid_index < 0)
    return skip("no routable RoCEv2 GID index for " + conn.dev_name());

  union ibv_gid gid;
  std::memcpy(gid.raw, conn.local().gid, 16);

  struct ibv_ah_attr aa;
  std::memset(&aa, 0, sizeof(aa));
  aa.is_global = 1;
  aa.port_num = 1;
  std::memcpy(aa.grh.dgid.raw, gid.raw, 16);
  aa.grh.sgid_index = static_cast<uint8_t>(sgid_index);
  aa.grh.hop_limit = 64;

  for (int i = 0; i < 64; ++i) {
    struct ibv_ah* ah = ibv_create_ah(conn.pd(), &aa);
    if (ah == nullptr)
      return fail("ibv_create_ah failed at iter " + std::to_string(i) + ": " +
                  std::strerror(errno));
    if (ibv_destroy_ah(ah) != 0)
      return fail("ibv_destroy_ah failed at iter " + std::to_string(i));
  }
  return pass();
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
