/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// HIP + Verbs mixed data-path tests (single-process / loopback group):
//   * vat_vmm_dmabuf_reg_not_regmr   — VMM buffers must register via dmabuf only
//   * vat_gpu_read_flush_after_recv  — GDR read-flush ordering (MUST-PASS)
//   * vat_gpu_mr_dereg_inflight      — MR teardown under RDMA load
//
// Each uses a loopback RC QP (self-connect) so a single process issues RDMA to
// its own GPU memory. The GPU-NIC coherence test (2-rank) lives in
// verbs/hip_coherence.cc; the alloc x reg matrix in verbs/hip_alloc_matrix.cc.
//
// Built only with BOTH HIP and ibverbs (see CMake). Without either, empty TU.

#if defined(ACCEPT_HAVE_HIP) && defined(ACCEPT_HAVE_IBVERBS)

#include "verbs/GdrMem.hpp"
#include "verbs/RcConn.hpp"
#include "verbs/VerbsPerf.hpp"

#include <unistd.h>  // close

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

constexpr size_t kMsgSize = 4096;
constexpr uint8_t kSeed = 0xA5;
constexpr int kPollTimeoutMs = 5000;

// Fill `buf` with the seeded byte pattern the RDMA verify step checks against.
void fill_seed_pattern(uint8_t* buf, size_t n) {
  for (size_t i = 0; i < n; ++i)
    buf[i] = static_cast<uint8_t>(kSeed + i);
}

// Allocate a device VMM buffer into the shared VmmMapping RAII owner (which
// unmaps/frees/releases on scope exit). Returns "" on success; a reason
// otherwise (VMM may be unsupported -> skip).
std::string vmm_alloc(VmmMapping& mapping, size_t req_size) {
  int dev = 0;
  hipError_t e = hipGetDevice(&dev);
  if (e != hipSuccess)
    return std::string("hipGetDevice: ") + hipGetErrorString(e);

  hipMemGenericAllocationHandle_t handle = 0;
  size_t size = 0;
  std::string create_err = vmm_create_pinned(req_size, dev, &handle, &size);
  if (!create_err.empty()) return create_err;

  // On failure the mapping owns `handle` (own_handle=true) and releases it on
  // scope exit, so no manual hipMemRelease here.
  return mapping.map(handle, size, dev, /*own_handle=*/true);
}

// Bring up a loopback RC QP (self-connect). Returns "" on success; sets skip=true
// when the failure is skip-worthy (no device / no active port).
std::string loopback_conn(verbsx::RcConn& conn, bool& skip_it) {
  skip_it = false;
  bool no_dev = false;
  std::string err = conn.open("", no_dev);
  if (!err.empty()) { skip_it = no_dev; return err; }
  err = conn.to_rts(conn.local());
  if (!err.empty()) return "loopback QP bring-up: " + err;
  return "";
}

// Build a RemoteBuf targeting `addr` with `rkey`.
verbsx::RemoteBuf remote_buf(void* addr, uint32_t rkey) {
  verbsx::RemoteBuf rb;
  rb.addr = reinterpret_cast<uint64_t>(addr);
  rb.rkey = rkey;
  return rb;
}

// Seed-fill a caller-owned host staging buffer and register it LOCAL_WRITE on
// `pd`. Returns the MR (nullptr on failure); `buf` must outlive the MR.
struct ibv_mr* register_staging_src(struct ibv_pd* pd,
                                    std::vector<uint8_t>& buf) {
  fill_seed_pattern(buf.data(), buf.size());
  return ibv_reg_mr(pd, buf.data(), buf.size(), IBV_ACCESS_LOCAL_WRITE);
}

// Post one signaled RDMA write of `payload_bytes` and reap its completion.
// Used by the functional teardown-stress loop, which needs distinct wr_ids.
std::string write_and_poll(verbsx::RcConn& connection, uint32_t local_key,
                           void* local_addr, size_t payload_bytes,
                           const verbsx::RemoteBuf& remote, uint64_t wr_id) {
  std::string reason = connection.post_write(
      local_key, local_addr, static_cast<uint32_t>(payload_bytes), remote,
      wr_id);
  if (reason.empty()) {
    reason = connection.poll(kPollTimeoutMs, nullptr);
  }
  return reason;
}

}  // namespace

// Confirms the RCCL rule that VMM/cuMem buffers must be registered with the NIC
// via dmabuf, NOT ibv_reg_mr: ibv_reg_mr on a VMM buffer must fail cleanly, the
// VMM state must survive that failure, and ibv_reg_dmabuf_mr + RDMA_WRITE must
// then succeed on the same buffer.
ACCEPT_TEST(vat_vmm_dmabuf_reg_not_regmr, "verbs/hip_mixed", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string err = loopback_conn(conn, skip_it);
  if (!err.empty()) return skip_it ? skip(err) : fail(err);

  VmmMapping vmm;
  const std::string alloc_err = vmm_alloc(vmm, kMsgSize);
  if (!alloc_err.empty()) return skip("VMM unavailable: " + alloc_err);

  // (1) ibv_reg_mr MUST fail on a VMM buffer (expected: Bad address / EFAULT).
  struct ibv_mr* bad =
      ibv_reg_mr(conn.pd(), vmm.get(), vmm.size(), verbsx::kRdmaAccessFlags);
  if (bad != nullptr) {
    ibvw::safe_dereg_mr(bad);
    return fail("ibv_reg_mr unexpectedly succeeded on a VMM buffer "
                "(RCCL requires dmabuf-only registration)");
  }

  // (2) VMM state must remain valid after the failed registration.
  hipError_t memset_err = hipMemset(vmm.get(), 0, vmm.size());
  if (memset_err != hipSuccess)
    return fail(std::string("VMM state corrupted after failed ibv_reg_mr "
                            "(hipMemset: ") +
                hipGetErrorString(memset_err) + ")");
  (void)hipDeviceSynchronize();

  // (3) dmabuf export + ibv_reg_dmabuf_mr MUST succeed. VMM VA ranges export via
  // hipMemGetHandleForAddressRange; the fd covers the range from vmm.get(), so
  // the MR offset is 0.
  int fd = -1;
  hipError_t export_err = hipMemGetHandleForAddressRange(
      &fd, reinterpret_cast<hipDeviceptr_t>(vmm.get()), vmm.size(),
      hipMemRangeHandleTypeDmaBufFd, 0);
  if (export_err != hipSuccess || fd < 0)
    return skip(std::string("VMM dmabuf export unsupported "
                            "(hipMemGetHandleForAddressRange: ") +
                hipGetErrorString(export_err) + ")");
  struct ibv_mr* mr = ibv_reg_dmabuf_mr(
      conn.pd(), 0, vmm.size(), reinterpret_cast<uint64_t>(vmm.get()), fd,
      verbsx::kRdmaAccessFlags);
  if (mr == nullptr) {
    ::close(fd);
    return fail(ibvw::errno_reason("ibv_reg_dmabuf_mr failed on VMM buffer"));
  }

  // (4) loopback RDMA_WRITE into the VMM buffer + verify.
  std::vector<uint8_t> staging(kMsgSize);
  struct ibv_mr* staging_mr = register_staging_src(conn.pd(), staging);

  PerfSink perf;
  std::string reason;
  bool ok = false;
  if (staging_mr == nullptr) {
    reason = "ibv_reg_mr(staging src) failed";
  } else {
    verbsx::RemoteBuf remote = remote_buf(vmm.get(), mr->rkey);
    reason = perf_rdma(perf, "vmm.dmabuf.write", conn, RdmaOp::Write,
                       staging_mr->lkey, staging.data(), kMsgSize, remote,
                       kPollTimeoutMs);
    if (reason.empty()) {
      std::string verify_err;
      ok = verbsx::verify_pattern(vmm.get(), kMsgSize, kSeed, &verify_err);
      if (!ok) reason = verify_err;
    }
  }

  ibvw::safe_dereg_mr(staging_mr);
  ibvw::safe_dereg_mr(mr);
  ::close(fd);

  if (!ok) return fail(reason.empty() ? "RDMA_WRITE verify failed" : reason);

  return pass_with_perf(std::move(perf));
}

// GDR read-flush ordering: after an RDMA_WRITE into a GPU dmabuf buffer, a
// loopback RDMA_READ on a SEPARATE flush QP (own PD) must complete before the
// buffer is read back, and the full payload must then be visible (no stale data).
ACCEPT_TEST(vat_gpu_read_flush_after_recv, "verbs/hip_mixed", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  verbsx::RcConn wconn;   // write QP
  verbsx::RcConn fconn;   // flush QP (separate PD)
  bool skip_it = false;
  std::string err = loopback_conn(wconn, skip_it);
  if (!err.empty()) return skip_it ? skip(err) : fail(err);
  err = loopback_conn(fconn, skip_it);
  if (!err.empty()) return skip_it ? skip(err) : fail(err);

  // GPU (fine-grained) target buffer, zeroed.
  std::string aerr;
  void* gpu = verbsx::alloc(verbsx::Alloc::FineGrain, kMsgSize, &aerr);
  if (gpu == nullptr) return skip("GPU alloc unsupported: " + aerr);
  HIP_CHECK(hipMemset(gpu, 0, kMsgSize));
  HIP_CHECK(hipDeviceSynchronize());

  std::string rerr;
  verbsx::Registration wmr = verbsx::reg(
      wconn.pd(), gpu, kMsgSize, verbsx::kRdmaAccessFlags, verbsx::Reg::DmabufV1,
      &rerr);
  if (wmr.mr == nullptr) {
    verbsx::free_(verbsx::Alloc::FineGrain, gpu);
    return skip("GPU dmabuf MR (write PD) unsupported: " + rerr);
  }
  verbsx::Registration fmr = verbsx::reg(
      fconn.pd(), gpu, kMsgSize, verbsx::kRdmaAccessFlags, verbsx::Reg::DmabufV1,
      &rerr);
  if (fmr.mr == nullptr) {
    verbsx::dereg(wmr);
    verbsx::free_(verbsx::Alloc::FineGrain, gpu);
    return skip("GPU dmabuf MR (flush PD) unsupported: " + rerr);
  }

  // Host staging source (pattern) for the write, and a flush sink for the read.
  std::vector<uint8_t> staging(kMsgSize);
  struct ibv_mr* staging_mr = register_staging_src(wconn.pd(), staging);
  std::vector<uint8_t> flush_sink(kMsgSize, 0);
  struct ibv_mr* flush_mr =
      ibv_reg_mr(fconn.pd(), flush_sink.data(), kMsgSize, IBV_ACCESS_LOCAL_WRITE);

  PerfSink perf;
  std::string reason;
  bool ok = false;
  if (staging_mr == nullptr || flush_mr == nullptr) {
    reason = "ibv_reg_mr(staging) failed";
  } else {
    verbsx::RemoteBuf write_target = remote_buf(gpu, wmr.mr->rkey);
    reason = perf_rdma(perf, "gdr.write", wconn, RdmaOp::Write, staging_mr->lkey,
                       staging.data(), kMsgSize, write_target, kPollTimeoutMs);

    if (reason.empty()) {
      // Read-flush: RDMA_READ the just-written GPU bytes over the flush QP.
      verbsx::RemoteBuf flush_target = remote_buf(gpu, fmr.mr->rkey);
      reason = perf_rdma(perf, "gdr.read_flush", fconn, RdmaOp::Read,
                         flush_mr->lkey, flush_sink.data(), kMsgSize,
                         flush_target, kPollTimeoutMs);
    }
    if (reason.empty()) {
      std::string verify_err;
      ok = verbsx::verify_pattern(gpu, kMsgSize, kSeed, &verify_err);
      if (!ok) reason = "stale GPU data after read-flush: " + verify_err;
    }
  }

  ibvw::safe_dereg_mr(staging_mr);
  ibvw::safe_dereg_mr(flush_mr);
  verbsx::dereg(fmr);
  verbsx::dereg(wmr);
  verbsx::free_(verbsx::Alloc::FineGrain, gpu);

  if (!ok) return fail(reason.empty() ? "read-flush verify failed" : reason);

  return pass_with_perf(std::move(perf));
}

// MR lifecycle / teardown under load: repeatedly allocate a GPU buffer, register
// it as a dmabuf MR, drive RDMA_WRITE traffic, then deregister and free in the
// correct order. Catches use-after-free / VMM corruption / teardown-ordering
// bugs (the isolated child + Runner watchdog catch a hang or crash).
ACCEPT_TEST(vat_gpu_mr_dereg_inflight, "verbs/hip_mixed", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  verbsx::RcConn conn;
  bool skip_it = false;
  std::string err = loopback_conn(conn, skip_it);
  if (!err.empty()) return skip_it ? skip(err) : fail(err);

  // Persistent host staging source.
  std::vector<uint8_t> src(kMsgSize);
  struct ibv_mr* smr = register_staging_src(conn.pd(), src);
  if (smr == nullptr) return fail("ibv_reg_mr(staging src) failed");

  constexpr int kIters = 100;
  std::string reason;
  bool skipped = false;

  // Perf: time each iteration's RDMA write in place (perf mode only) so the metric
  // comes from the real teardown-under-load loop, not a separate synthetic pass.
  std::vector<double> write_samples_us;
  for (int it = 0; it < kIters; ++it) {
    std::string aerr;
    void* gpu = verbsx::alloc(verbsx::Alloc::DeviceMalloc, kMsgSize, &aerr);
    if (gpu == nullptr) { reason = "GPU alloc failed: " + aerr; break; }

    std::string rerr;
    verbsx::Registration r = verbsx::reg(
        conn.pd(), gpu, kMsgSize, verbsx::kRdmaAccessFlags, verbsx::Reg::DmabufV1,
        &rerr);
    if (r.mr == nullptr) {
      verbsx::free_(verbsx::Alloc::DeviceMalloc, gpu);
      if (it == 0) { skipped = true; reason = "GPU dmabuf MR unsupported: " + rerr; }
      else reason = "GPU dmabuf MR failed mid-run: " + rerr;
      break;
    }

    verbsx::RemoteBuf rb = remote_buf(gpu, r.mr->rkey);
    const PerfClock::time_point write_start = PerfClock::now();
    reason = write_and_poll(conn, smr->lkey, src.data(), kMsgSize, rb,
                            static_cast<uint64_t>(it) + 1);
    if (reason.empty() && perf_probe_mode()) {
      write_samples_us.push_back(
          std::chrono::duration<double, std::micro>(PerfClock::now() -
                                                    write_start)
              .count());
    }

    // Spot-check payload integrity on the first and last iteration (not every
    // one, to keep the teardown loop cheap) -- the completion alone does not
    // prove the bytes landed.
    if (reason.empty() && (it == 0 || it == kIters - 1)) {
      std::string verr;
      if (!verbsx::verify_pattern(gpu, kMsgSize, kSeed, &verr))
        reason = "payload mismatch: " + verr;
    }

    // Teardown ordering under load: dereg MR (+ close dmabuf) BEFORE freeing the
    // HIP allocation the MR was pinning.
    verbsx::dereg(r);
    verbsx::free_(verbsx::Alloc::DeviceMalloc, gpu);
    if (!reason.empty()) { reason = "iter " + std::to_string(it) + ": " + reason; break; }
  }

  ibvw::safe_dereg_mr(smr);
  if (skipped) return skip(reason);
  if (!reason.empty()) return fail(reason);

  PerfSink perf;
  emit_latency_samples(perf, "write_during_dereg.lat", std::move(write_samples_us));
  return pass_with_perf(std::move(perf));
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP && ACCEPT_HAVE_IBVERBS
