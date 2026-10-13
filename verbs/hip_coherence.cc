/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// vat_gpu_nic_coherence_streamwait -- GPU/NIC PCIe coherence (from the
// GPU/NIC coherence reference program). Two ranks: rank 0 owns a fine-grained GPU "signal" and
// arms hipStreamWaitValue64 on it; rank 1's NIC writes the signal over RDMA.
// The GPU must observe the NIC's PCIe write (path B) even when its L2 was warmed
// with the stale value via the shader path (path A).
//
// Semantics are INVERTED vs. the reproducers: in the repro a hang == bug
// CONFIRMED; here a hang/timeout == FAIL and a completed stream wait == PASS
// (the coherence bug is NOT present).
//
// Three variants:
//   wait_first  -- GPU wait armed BEFORE the NIC write (baseline ordering).
//   put_first   -- GPU pre-reads the signal (warms L2 stale), NIC writes, THEN
//                  the wait is armed -- the hard coherence case.
//   cross_nic   -- put_first ordering but driven through a NIC that is NOT
//                  GPU-0's PCIe-nearest device (resolved by name via
//                  far_nic_for_gpu); skips when the host has no such NIC.
//
// TwoNode topology: runs in-process in each worker under the shared
// SocketBootstrap; excluded from single-node runs. Built only with BOTH HIP and
// ibverbs; otherwise an empty TU.

#if defined(ACCEPT_HAVE_HIP) && defined(ACCEPT_HAVE_IBVERBS)

#include "verbs/GdrMem.hpp"
#include "verbs/RcConn.hpp"
#include "verbs/TwoNodeGate.hpp"
#include "verbs/VerbsPerf.hpp"

#include <cstdint>
#include <string>

#include "common/Bootstrap.hpp"
#include "common/Capability.hpp"
#include "common/Perf.hpp"
#include "common/PosixUtil.hpp"
#include "common/TestRegistry.hpp"
#include "hip/utils/HipModule.hpp"

namespace acceptance {

namespace {

using verbsx::Gate;
using verbsx::reconcile_gate;

constexpr uint64_t kSignalTarget = 1;
constexpr int kOwnerRank = 0;  // arms the GPU stream wait; the other rank writes
constexpr int kWaitTimeoutMs = 8000;
constexpr size_t kSignalBufSize = 4096;  // fine-grained GPU signal window
constexpr int kStreamPollSleepMs = 1;    // stream_wait_timeout poll cadence
// hipStreamWaitValue64 compares (value & mask) >= target; all-ones matches every
// bit, so any store >= target wakes the wait.
constexpr uint64_t kStreamWaitMask = 0xFFFFFFFFFFFFFFFFULL;
// Sentinel the owner pre-loads into the read-back sink; if it survives, the GPU
// never actually read the (stale 0) signal and the hazard was not established.
constexpr uint64_t kStaleReadPoison = 0xEEEEEEEEEEEEEEEEULL;

enum class Variant { WaitFirst, PutFirst, CrossNic };

// Pick an RDMA device that is NOT GPU `gpu`'s PCIe-nearest NIC (the "far NIC"
// path the cross_nic variant is meant to stress). Uses the shared affinity
// probe. Returns "" when GPU->NIC affinity is unknown or there is no other
// device (single NIC), so the variant skips instead of silently reusing the
// nearest NIC.
std::string far_nic_for_gpu(int gpu) {
  const EnvironmentInfo& env = environment();
  std::string nearest;
  for (const auto& a : env.gpu_nic_affinity)
    if (a.gpu_ordinal == gpu) { nearest = a.nearest_nic; break; }
  if (nearest.empty()) return "";  // can't tell which NIC is "far"
  for (const auto& d : env.rdma_devices)
    if (!d.device.empty() && d.device != nearest) return d.device;
  return "";
}

// Poll a stream to completion within timeout_ms. true == completed.
bool stream_wait_timeout(hipStream_t s, int timeout_ms) {
  const long timeout_ms_l = static_cast<long>(timeout_ms);
  for (long waited = 0;; waited += kStreamPollSleepMs) {
    hipError_t q = hipStreamQuery(s);
    if (q == hipSuccess) return true;
    if (q != hipErrorNotReady) return false;
    if (waited >= timeout_ms_l) return false;
    posix_util::sleep_ms(kStreamPollSleepMs);
  }
}

// Owner-side resources, released together on every exit path (the writer rank
// owns only src_mr, torn down separately).
struct OwnerResources {
  verbsx::Registration sig_reg;  // signal-window MR
  void* signal = nullptr;        // fine-grained GPU signal window
  void* sink = nullptr;          // pre-read sink (put_first only)
  hipStream_t stream = nullptr;
};

void owner_release(OwnerResources& r) {
  verbsx::dereg(r.sig_reg);
  if (r.sink) verbsx::free_(verbsx::Alloc::DeviceMalloc, r.sink);
  if (r.signal) verbsx::free_(verbsx::Alloc::FineGrain, r.signal);
  if (r.stream) (void)hipStreamDestroy(r.stream);
  r.sink = r.signal = nullptr;
  r.stream = nullptr;
}

// Read the signal back to the host and confirm the fresh value landed. Returns
// "" when signal >= target, else a reason (read-back error or a false wake).
std::string verify_signal_landed(void* signal) {
  uint64_t landed = 0;
  hipError_t copy_err =
      hipMemcpy(&landed, signal, sizeof(landed), hipMemcpyDeviceToHost);
  if (copy_err != hipSuccess)
    return std::string("signal read-back hipMemcpy(D2H): ") +
           hipGetErrorString(copy_err);
  if (landed < kSignalTarget)
    return "stream wait fired but signal=" + std::to_string(landed) +
           " < target " + std::to_string(kSignalTarget) + " (false wake)";
  return "";
}

// Arm the GPU stream wait on `signal`. Returns hipSuccess, or the error the
// caller maps to a skip when the runtime lacks hipStreamWaitValue64.
hipError_t arm_signal_wait(hipStream_t stream, void* signal) {
  return hipStreamWaitValue64(stream, signal, kSignalTarget,
                              hipStreamWaitValueGte, kStreamWaitMask);
}

// Poll the already-armed wait to completion within the timeout, then verify the
// landed value. `timeout_msg` describes the coherence/hang failure. Emits the
// GPU signal-observation latency (arm -> stream completes) into `perf`.
TestResult poll_and_verify(hipStream_t stream, void* signal,
                           const std::string& timeout_msg, PerfSink& perf) {
  PerfLoop observe(kSinglePass);
  bool completed = false;
  for (auto pass : observe) {
    completed = stream_wait_timeout(stream, kWaitTimeoutMs);
  }
  if (!completed) {
    return fail(timeout_msg);
  }
  observe.emit_latency(perf, "nic_to_gpu_observe.lat");

  const std::string landed = verify_signal_landed(signal);
  return landed.empty() ? pass() : fail(landed);
}

// Writer NIC put: RDMA-write the signal value into the owner's window and reap
// the completion, emitting the put latency into `perf`. Returns "" or a reason.
std::string writer_put_signal(verbsx::RcConn& conn, uint32_t lkey,
                              uint64_t* value, const verbsx::RemoteBuf& rb,
                              PerfSink& perf) {
  return perf_rdma(perf, "coherence.nic_put", conn, RdmaOp::Write, lkey, value,
                   sizeof(*value), rb, kWaitTimeoutMs, kSinglePass);
}

// Outcome of the owner's put_first pre-read warm-up.
struct PreReadOutcome {
  bool ok = false;
  bool is_skip = false;  // when !ok: true -> skip, false -> fail
  std::string reason;
};

// put_first warm-up: a single-thread kernel reads the (stale 0) signal into the
// sink, warming the GPU L2 with the stale value; then confirm the sink really
// holds 0 so a later "pass" cannot be a coincidence. Does no barriers.
PreReadOutcome owner_pre_read(hipStream_t stream, void* signal, void* sink) {
  struct ReadArgs {  // k_read_u64(const uint64_t* signal, uint64_t* sink)
    const uint64_t* signal;
    uint64_t* sink;
  } args{reinterpret_cast<const uint64_t*>(signal),
         reinterpret_cast<uint64_t*>(sink)};
  std::string kerr;
  if (hip_kernel_launch("k_read_u64", dim3(1), dim3(1), 0, stream, &args,
                        sizeof(args), &kerr) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess)
    return {false, true, "pre-read kernel unavailable: " + kerr};
  uint64_t stale = kStaleReadPoison;
  if (hipMemcpy(&stale, sink, sizeof(stale), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return {false, true, "pre-read sink readback failed"};
  if (stale != 0)
    return {false, false,
            "pre-read did not observe stale 0 (saw " + std::to_string(stale) +
                "); coherence hazard not established"};
  return {true, false, ""};
}

// Phase-4 handles for the ordered coherence sequence: owner drives stream/signal,
// writer drives conn/src_mr; peer_rb is the owner's exchanged signal window.
struct Phase4Ctx {
  Bootstrap* boot = nullptr;
  bool is_owner = false;
  hipStream_t stream = nullptr;  // owner
  void* signal = nullptr;        // owner
  void* sink = nullptr;          // owner (put_first)
  verbsx::RcConn* conn = nullptr;      // writer
  struct ibv_mr* src_mr = nullptr;     // writer
  uint64_t* writer_val = nullptr;      // writer
  verbsx::RemoteBuf peer_rb;           // writer target
  PerfSink* perf = nullptr;            // per-rank perf metrics sink
};

// wait_first: the owner arms the wait BEFORE the NIC write (barrier A gates the
// write on the armed wait). Baseline ordering. Barriers: A, B.
TestResult run_wait_first(const Phase4Ctx& c) {
  if (c.is_owner) {
    hipError_t wait_err = arm_signal_wait(c.stream, c.signal);
    if (wait_err != hipSuccess) {
      c.boot->barrier();  // A
      c.boot->barrier();  // B (keep matched with the writer)
      return skip(std::string("hipStreamWaitValue64 unsupported: ") +
                  hipGetErrorString(wait_err));
    }
    c.boot->barrier();  // A: wait armed
    TestResult result =
        poll_and_verify(c.stream, c.signal,
                        "GPU did not observe NIC write within " +
                            std::to_string(kWaitTimeoutMs) + "ms (coherence/hang)",
                        *c.perf);
    c.boot->barrier();  // B: done
    return result;
  }
  c.boot->barrier();  // A: wait until owner armed
  std::string put_err = writer_put_signal(*c.conn, c.src_mr->lkey, c.writer_val,
                                          c.peer_rb, *c.perf);
  c.boot->barrier();  // B
  return put_err.empty() ? pass() : fail("NIC write: " + put_err);
}

// put_first / cross_nic: the owner pre-reads (warming L2 with the stale value),
// the writer puts, THEN the owner arms the wait -- the hard coherence case.
// Barriers: A, B, C.
TestResult run_put_first(const Phase4Ctx& c) {
  if (c.is_owner) {
    PreReadOutcome pr = owner_pre_read(c.stream, c.signal, c.sink);
    if (!pr.ok) {
      c.boot->barrier();  // A
      c.boot->barrier();  // B
      c.boot->barrier();  // C (keep matched with the writer)
      return pr.is_skip ? skip(pr.reason) : fail(pr.reason);
    }
    c.boot->barrier();  // A: pre-read done, writer may put
    c.boot->barrier();  // B: writer put landed
    hipError_t wait_err = arm_signal_wait(c.stream, c.signal);
    TestResult result =
        (wait_err != hipSuccess)
            ? skip(std::string("hipStreamWaitValue64 unsupported: ") +
                   hipGetErrorString(wait_err))
            : poll_and_verify(
                  c.stream, c.signal,
                  "GPU did not observe NIC write after put_first warm-up within " +
                      std::to_string(kWaitTimeoutMs) + "ms (coherence/hang)",
                  *c.perf);
    c.boot->barrier();  // C
    return result;
  }
  c.boot->barrier();  // A: owner pre-read done
  std::string put_err = writer_put_signal(*c.conn, c.src_mr->lkey, c.writer_val,
                                          c.peer_rb, *c.perf);
  c.boot->barrier();  // B: put landed
  c.boot->barrier();  // C
  return put_err.empty() ? pass() : fail("NIC write: " + put_err);
}

TestResult coherence_run(TestContext& ctx, Variant variant) {
  if (ctx.size != 2 || ctx.boot == nullptr) return skip("needs 2 ranks");
  Bootstrap* boot = ctx.boot;
  const bool is_owner = (ctx.rank == kOwnerRank);
  const bool put_first = (variant != Variant::WaitFirst);
  const int peer_rank = 1 - ctx.rank;

  std::string reason;

  // ---- Phase 1: device + QP + buffers ----
  verbsx::RcConn conn;
  OwnerResources owner;                 // owner-only resources
  uint64_t writer_val = kSignalTarget;  // writer: host source holding the value
  struct ibv_mr* src_mr = nullptr;      // writer-only

  // Release whatever this rank owns; called on every gated early return and once
  // at the end (owner uses the bundle, writer just its src MR).
  auto release_local = [&]() {
    if (is_owner)
      owner_release(owner);
    else
      ibvw::safe_dereg_mr(src_mr);
  };

  // cross_nic: bind the QP to a NIC that is NOT GPU-0's PCIe-nearest device so
  // the GDR path crosses a PCIe switch / root complex (the coherence hazard the
  // variant targets). wait_first/put_first use the first active device ("").
  std::string dev_name;
  Gate mine = Gate::Ok;
  if (variant == Variant::CrossNic) {
    dev_name = far_nic_for_gpu(0);
    if (dev_name.empty()) {
      mine = Gate::Skip;
      reason = "cross_nic: no RDMA device other than GPU-0's nearest NIC";
    }
  }

  if (mine == Gate::Ok) {
    bool no_dev = false;
    std::string open_err = conn.open(dev_name, no_dev);
    if (!open_err.empty()) {
      mine = no_dev ? Gate::Skip : Gate::Fail;
      reason = open_err;
    }
  }

  if (mine == Gate::Ok && is_owner) {
    std::string alloc_err;
    owner.signal =
        verbsx::alloc(verbsx::Alloc::FineGrain, kSignalBufSize, &alloc_err);
    if (owner.signal == nullptr) {
      mine = Gate::Skip;
      reason = "GPU signal alloc: " + alloc_err;
    } else if (hipMemset(owner.signal, 0, kSignalBufSize) != hipSuccess ||
               hipDeviceSynchronize() != hipSuccess) {
      mine = Gate::Fail;
      reason = "signal init failed";
    }
    if (mine == Gate::Ok) {
      std::string reg_err;
      owner.sig_reg =
          verbsx::reg(conn.pd(), owner.signal, kSignalBufSize,
                      verbsx::kRdmaAccessFlags, verbsx::Reg::DmabufV1, &reg_err);
      if (owner.sig_reg.mr == nullptr) {
        mine = Gate::Skip;
        reason = "signal dmabuf MR: " + reg_err;
      }
    }
    if (mine == Gate::Ok && put_first) {
      std::string sink_err;
      owner.sink =
          verbsx::alloc(verbsx::Alloc::DeviceMalloc, sizeof(uint64_t), &sink_err);
      if (owner.sink == nullptr) {
        mine = Gate::Skip;
        reason = "pre-read sink alloc: " + sink_err;
      }
    }
    if (mine == Gate::Ok && hipStreamCreate(&owner.stream) != hipSuccess) {
      mine = Gate::Fail;
      reason = "hipStreamCreate failed";
    }
  }

  if (mine == Gate::Ok && !is_owner) {
    src_mr = ibv_reg_mr(conn.pd(), &writer_val, sizeof(writer_val),
                        IBV_ACCESS_LOCAL_WRITE);
    if (src_mr == nullptr) {
      mine = Gate::Fail;
      reason = "writer src reg_mr failed";
    }
  }

  if (Gate gate = reconcile_gate(boot, mine); gate != Gate::Ok) {
    release_local();
    const std::string r = reason.empty() ? "peer not ready" : reason;
    return gate == Gate::Skip ? skip(r) : fail(r);
  }

  // ---- Phase 2: exchange QP info + connect ----
  verbsx::QpInfo local = conn.local();
  verbsx::QpInfo remote;
  boot->exchange(peer_rank, &local, &remote, sizeof(verbsx::QpInfo));
  {
    std::string connect_err = conn.to_rts(remote);
    mine = connect_err.empty() ? Gate::Ok : Gate::Fail;
    if (!connect_err.empty()) reason = "connect: " + connect_err;
    if (Gate gate = reconcile_gate(boot, mine); gate != Gate::Ok) {
      release_local();
      return gate == Gate::Skip
                 ? skip(reason)
                 : fail(reason.empty() ? "peer connect failed" : reason);
    }
  }

  // ---- Phase 3: exchange the signal window (owner -> writer) ----
  verbsx::RemoteBuf my_rb;
  if (is_owner) {
    my_rb.addr = reinterpret_cast<uint64_t>(owner.signal);
    my_rb.rkey = owner.sig_reg.mr->rkey;
  }
  verbsx::RemoteBuf peer_rb;
  boot->exchange(peer_rank, &my_rb, &peer_rb, sizeof(verbsx::RemoteBuf));

  // ---- Phase 4: ordered coherence sequence ----
  PerfSink perf;
  Phase4Ctx phase4;
  phase4.boot = boot;
  phase4.is_owner = is_owner;
  phase4.stream = owner.stream;
  phase4.signal = owner.signal;
  phase4.sink = owner.sink;
  phase4.conn = &conn;
  phase4.src_mr = src_mr;
  phase4.writer_val = &writer_val;
  phase4.peer_rb = peer_rb;
  phase4.perf = &perf;

  boot->barrier();  // both connected & ready
  TestResult result = put_first ? run_put_first(phase4) : run_wait_first(phase4);

  release_local();

  // Only fold perf metrics into a passing verdict; skip/fail keep their reason.
  if (result.status == Status::Pass) {
    result.perf = std::move(perf);
  }
  return result;
}

}  // namespace

ACCEPT_TEST(vat_gpu_nic_coherence_streamwait_wait_first, "verbs/hip_mixed",
            Topology::TwoNode, Kind::Functional, 60) {
  return acceptance::coherence_run(ctx, Variant::WaitFirst);
}

ACCEPT_TEST(vat_gpu_nic_coherence_streamwait_put_first, "verbs/hip_mixed",
            Topology::TwoNode, Kind::Functional, 60) {
  return acceptance::coherence_run(ctx, Variant::PutFirst);
}

ACCEPT_TEST(vat_gpu_nic_coherence_streamwait_cross_nic, "verbs/hip_mixed",
            Topology::TwoNode, Kind::Functional, 60) {
  return acceptance::coherence_run(ctx, Variant::CrossNic);
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP && ACCEPT_HAVE_IBVERBS
