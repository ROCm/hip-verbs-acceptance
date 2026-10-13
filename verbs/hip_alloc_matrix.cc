/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// vat_hipalloc_x_ibvreg -- HIP allocation modes x verbs registration methods
// capability matrix (from the HIP-allocation reference program). Each of the 7x3=21
// combinations is a separate registered test (one CI line each).
//
// This mirrors the reference program EXACTLY: a two-endpoint client/server pair
// (Topology::TwoNode) with a real RDMA_WRITE_WITH_IMM transfer, rather than a
// loopback self-connect. Per (alloc x reg) case, on both ranks:
//   1. bring up an RC QP and connect to the peer (QpInfo over Bootstrap),
//   2. allocate a 4 KiB buffer with the allocation mode; handshake alloc_ok --
//      if EITHER side cannot allocate, both SKIP,
//   3. register it with the registration method; handshake reg_ok -- if EITHER
//      side cannot register, both SKIP,
//   4. rank 0 = server (receiver): zero the buffer, post a recv WR for the imm
//      notification, publish its {addr,rkey}; rank 1 = client (sender): fill the
//      pattern and RDMA_WRITE_WITH_IMM into the server window,
//   5. server waits for IBV_WC_RECV_RDMA_WITH_IMM and verifies the bytes; the
//      verdict is unified across both ranks.
// A data mismatch or RDMA error fails; an unsupported alloc/reg combination
// skips cleanly (e.g. dmabuf export of host/managed memory, which HSA rejects).
//
// Built only with BOTH HIP and ibverbs (see CMake). Without either, empty TU.

#if defined(ACCEPT_HAVE_HIP) && defined(ACCEPT_HAVE_IBVERBS)

#include "verbs/GdrMem.hpp"
#include "verbs/RcConn.hpp"
#include "verbs/TwoNodeGate.hpp"
#include "verbs/VerbsPerf.hpp"

#include <cstdint>
#include <cstring>
#include <string>

#include "common/Bootstrap.hpp"
#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {

namespace {

using verbsx::Gate;
using verbsx::reconcile_gate;

constexpr size_t   kMsgSize      = 4096;
constexpr uint8_t  kSeed         = 0x5A;
constexpr uint32_t kImmDone      = 0x0000BEEF;  // imm payload (value unused)
constexpr int      kPollTimeoutMs = 5000;
constexpr int      kServerRank   = 0;

TestResult matrix_case(TestContext& ctx, verbsx::Alloc amode, verbsx::Reg rmode) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_server = (ctx.rank == kServerRank);

  verbsx::RcConn conn;
  void* buf = nullptr;
  verbsx::Registration rreg;
  uint8_t* recv_buf = nullptr;         // server: imm-notification recv target
  struct ibv_mr* recv_mr = nullptr;
  std::string reason;
  PerfSink perf;

  // ---- Phase 1: open device + QP (INIT) ------------------------------------
  Gate mine = Gate::Ok;
  bool no_dev = false;
  std::string open_err = conn.open("", no_dev);
  if (!open_err.empty()) {
    mine = no_dev ? Gate::Skip : Gate::Fail;
    reason = open_err;
  }
  if (Gate gate = reconcile_gate(boot, mine); gate != Gate::Ok)
    return gate == Gate::Skip
               ? skip(reason.empty() ? "peer has no RDMA device" : reason)
               : fail(reason.empty() ? "peer QP open failed" : reason);

  // ---- Phase 2: exchange QpInfo + connect to peer (INIT->RTR->RTS) ----------
  {
    verbsx::QpInfo local = conn.local();
    verbsx::QpInfo remote;
    boot->exchange(1 - ctx.rank, &local, &remote, sizeof(verbsx::QpInfo));
    std::string connect_err = conn.to_rts(remote);
    Gate connect_gate = connect_err.empty() ? Gate::Ok : Gate::Fail;
    if (!connect_err.empty()) reason = "connect: " + connect_err;
    if (Gate gate = reconcile_gate(boot, connect_gate); gate != Gate::Ok)
      return fail(reason.empty() ? "peer connect failed" : reason);
  }

  // ---- Phase 3: allocate + alloc_ok handshake ------------------------------
  std::string alloc_err;
  buf = verbsx::alloc(amode, kMsgSize, &alloc_err);
  uint8_t alloc_ok = buf ? 1 : 0;
  uint8_t peer_alloc = 0;
  boot->exchange(1 - ctx.rank, &alloc_ok, &peer_alloc, 1);
  if (!alloc_ok || !peer_alloc) {
    if (buf) verbsx::free_(amode, buf);
    return skip(std::string("alloc ") + verbsx::alloc_name(amode) +
                (alloc_ok ? " unsupported on peer"
                          : (" unsupported: " + alloc_err)));
  }

  // ---- Phase 3b: register + reg_ok handshake -------------------------------
  std::string reg_err;
  rreg = verbsx::reg(conn.pd(), buf, kMsgSize, verbsx::kRdmaAccessFlags, rmode,
                     &reg_err);
  uint8_t reg_ok = rreg.mr ? 1 : 0;
  uint8_t peer_reg = 0;
  boot->exchange(1 - ctx.rank, &reg_ok, &peer_reg, 1);
  if (!reg_ok || !peer_reg) {
    verbsx::dereg(rreg);
    verbsx::free_(amode, buf);
    return skip(std::string("register ") + verbsx::reg_name(rmode) + " for " +
                verbsx::alloc_name(amode) +
                (reg_ok ? " unsupported on peer" : (" unsupported: " + reg_err)));
  }

  // ---- Phase 4: RDMA_WRITE_WITH_IMM (server=receiver, client=sender) --------
  Gate data_gate = Gate::Ok;
  std::string data_err;

  if (is_server) {
    // Zero so stale bytes are detectable, then arm a recv for the imm.
    if (hipMemset(buf, 0, kMsgSize) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
      data_gate = Gate::Fail;
      data_err = "server buffer zero failed";
    }
    if (data_gate == Gate::Ok) {
      recv_buf = new uint8_t[64];
      recv_mr = ibv_reg_mr(conn.pd(), recv_buf, 64, IBV_ACCESS_LOCAL_WRITE);
      if (recv_mr == nullptr) {
        data_gate = Gate::Fail;
        data_err = "server recv_mr reg failed";
      }
    }
    if (data_gate == Gate::Ok) {
      std::string post_err =
          conn.post_recv(recv_mr->lkey, recv_buf, 64, /*wr_id=*/1000);
      if (!post_err.empty()) {
        data_gate = Gate::Fail;
        data_err = "server post_recv: " + post_err;
      }
    }
  }

  // Publish the server's target window to the client.
  verbsx::RemoteBuf my_rb;
  if (is_server) { my_rb.addr = reinterpret_cast<uint64_t>(buf); my_rb.rkey = rreg.mr->rkey; }
  verbsx::RemoteBuf peer_rb;
  boot->exchange(1 - ctx.rank, &my_rb, &peer_rb, sizeof(verbsx::RemoteBuf));

  boot->barrier();  // server recv is posted before the client writes

  if (is_server) {
    if (data_gate == Gate::Ok) {
      struct ibv_wc wc;
      std::memset(&wc, 0, sizeof(wc));

      PerfLoop recv_complete(kSinglePass);
      std::string poll_err;
      for (auto pass : recv_complete) {
        poll_err = conn.poll(kPollTimeoutMs, &wc);
      }

      if (!poll_err.empty()) {
        data_gate = Gate::Fail;
        data_err = "server recv completion: " + poll_err;
      } else if (wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
        data_gate = Gate::Fail;
        data_err = "server: unexpected opcode " + std::to_string(wc.opcode);
      } else {
        recv_complete.emit_latency(perf, "transfer.lat");
        std::string verify_err;
        if (!verbsx::verify_pattern(buf, kMsgSize, kSeed, &verify_err)) {
          data_gate = Gate::Fail;
          data_err = "server verify: " + verify_err;
        }
      }
    }
  } else {
    std::string fill_err;
    if (!verbsx::fill_pattern(buf, kMsgSize, kSeed, &fill_err)) {
      data_gate = Gate::Fail;
      data_err = "client fill: " + fill_err;
    }
    if (data_gate == Gate::Ok) {
      PerfLoop write_imm(kSinglePass);
      std::string write_err;
      for (auto pass : write_imm) {
        write_err = conn.post_write_imm(rreg.mr->lkey, buf,
                                        static_cast<uint32_t>(kMsgSize), peer_rb,
                                        kImmDone, /*wr_id=*/1);
        if (write_err.empty()) {
          write_err = conn.poll(kPollTimeoutMs, nullptr);
        }
      }
      if (!write_err.empty()) {
        data_gate = Gate::Fail;
        data_err = "client write_imm: " + write_err;
      } else {
        write_imm.emit_latency(perf, "alloc_matrix.write_imm");
      }
    }
  }

  const Gate gate = reconcile_gate(boot, data_gate);

  // ---- teardown ----
  if (recv_mr) ibvw::safe_dereg_mr(recv_mr);
  delete[] recv_buf;
  verbsx::dereg(rreg);
  verbsx::free_(amode, buf);

  if (gate != Gate::Ok)
    return fail(data_err.empty() ? "peer data-path failed" : data_err);

  return pass_with_perf(std::move(perf));
}

}  // namespace

// One registered test per (alloc x reg) combination. Ids read
// vat_hipalloc_x_ibvreg__<alloc_tag>__<reg_tag>.
#define VAT_MATRIX(SUF, A, R)                                                     \
  ACCEPT_TEST(vat_hipalloc_x_ibvreg__##SUF, "verbs/hip_mixed", Topology::TwoNode, \
              Kind::Functional, 30) {                                             \
    return acceptance::matrix_case(ctx, verbsx::Alloc::A, verbsx::Reg::R);        \
  }

VAT_MATRIX(device__regmr,            DeviceMalloc,    Std)
VAT_MATRIX(device__dmabuf_v1,        DeviceMalloc,    DmabufV1)
VAT_MATRIX(device__dmabuf_v2_pcie,   DeviceMalloc,    DmabufV2Pcie)

VAT_MATRIX(host_default__regmr,          HostDefault, Std)
VAT_MATRIX(host_default__dmabuf_v1,      HostDefault, DmabufV1)
VAT_MATRIX(host_default__dmabuf_v2_pcie, HostDefault, DmabufV2Pcie)

VAT_MATRIX(host_wc__regmr,          HostWC, Std)
VAT_MATRIX(host_wc__dmabuf_v1,      HostWC, DmabufV1)
VAT_MATRIX(host_wc__dmabuf_v2_pcie, HostWC, DmabufV2Pcie)

VAT_MATRIX(host_coherent__regmr,          HostCoherent, Std)
VAT_MATRIX(host_coherent__dmabuf_v1,      HostCoherent, DmabufV1)
VAT_MATRIX(host_coherent__dmabuf_v2_pcie, HostCoherent, DmabufV2Pcie)

VAT_MATRIX(host_noncoherent__regmr,          HostNonCoherent, Std)
VAT_MATRIX(host_noncoherent__dmabuf_v1,      HostNonCoherent, DmabufV1)
VAT_MATRIX(host_noncoherent__dmabuf_v2_pcie, HostNonCoherent, DmabufV2Pcie)

VAT_MATRIX(finegrain__regmr,          FineGrain, Std)
VAT_MATRIX(finegrain__dmabuf_v1,      FineGrain, DmabufV1)
VAT_MATRIX(finegrain__dmabuf_v2_pcie, FineGrain, DmabufV2Pcie)

VAT_MATRIX(managed__regmr,          Managed, Std)
VAT_MATRIX(managed__dmabuf_v1,      Managed, DmabufV1)
VAT_MATRIX(managed__dmabuf_v2_pcie, Managed, DmabufV2Pcie)

#undef VAT_MATRIX

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP && ACCEPT_HAVE_IBVERBS
