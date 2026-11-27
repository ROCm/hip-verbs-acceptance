/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Verbs acceptance test -- ibv_post_send cost (vat_post_send_cost).
//
// What this guards. On ionic (AMD Pensando) the send queue can live in host
// memory or in the NIC's controller memory buffer (CMB). With an UNCACHED CMB
// mapping every ibv_post_send copies the WQE with posted MMIO stores and costs
// ~3 us of CPU per call -- 10-20x a host-memory SQ -- independent of QP count and
// WQE shape. RCCL posts one WQE per proxy step, so that cost lands on the
// critical path of small collectives. The same hardware with a host-memory SQ or
// a write-combining CMB mapping posts in ~0.15-0.5 us. Other NICs have no such
// placement choice but the same kind of regression (a doorbell/WQE mapping that
// silently stops being write-combining) shows up as a post that is several times
// slower than normal.
//
// The test ports perf-logs/cmb-post-cost/postlat.c: loopback RC QPs (several QP
// counts and WQE shapes), every ibv_post_send timed in a tight, completion-
// throttled loop. It measures the OUT-OF-BOX configuration first (IONIC_SQ_CMB
// unset: what an application that does nothing gets) and then judges it:
//
//   * ionic hardware and libionic loadable (ionicdv::init()): the same shapes are
//     measured with the non-default SQ placements -- IONIC_SQ_CMB=0 (host
//     memory), the PD API ionic_dv_pd_set_sqcmb(pd, false, ...) (what
//     RCCL_AINIC_SQ_CMB=0 does), IONIC_SQ_CMB=ew and =xw (CMB, write-combining
//     mapping) -- and the default is BROKEN when any alternative beats it by more
//     than the tolerance (default 30%, ACCEPT_POST_SEND_TOLERANCE_PCT), judged on
//     the 95% CI of the median so noise cannot trip it. Alternatives that are
//     slower than the default are reported, never failed: the default is the
//     thing under test.
//   * anything else (no libionic, or a non-ionic NIC such as Broadcom Thor or
//     ConnectX): there is nothing to compare against, so the default is compared
//     with the hardcoded expected median for that NIC family (postcost::
//     reference_ns_for, ACCEPT_POST_SEND_REF_NS) grown by the same tolerance; it
//     is broken when even the low end of its median CI is above that line.
//
// Only a broken default fails the test, and only then does the message carry a
// recommendation. A per-shape table of every measured configuration is printed to
// stderr either way. Alternatives that do not apply on a host (a host-memory knob
// that is ignored, a CMB placement that does not land in the CMB) are skipped
// with a note.
//
// Perf mode (--kind perf) adds the per-config post-cost samples and the post->CQE
// latency of a single signalled WR as report-only metrics.
//
// Method notes. The cost is measured per BLOCK of 64 back-to-back ibv_post_send
// calls (WRs are built before the clock starts, CQ reaping happens between
// blocks) and reported per call; the verdict uses the median over 512 blocks.
// Timing single calls is not enough: an uncached MMIO store retires into the
// CPU's store buffer immediately, and the stall is paid by the next call, so a
// per-call timestamp pair around a UC-CMB post reads ~40 ns while the loop really
// costs ~3.0 us per post (measured on TW: 40 ns per call vs 3030 ns per post
// over the whole loop, same as postlat.c's rdtscp+lfence timing). A block
// absorbs the deferred stall in all but its last call. Signalling every 8th WR
// mirrors postlat.c / RCCL's batching so the SQ never wraps; a signalled marker
// per QP drains every WR before the QPs are destroyed.
//
// Built only with ACCEPT_WITH_IBVERBS=ON (ACCEPT_HAVE_IBVERBS); no libibverbs-dev
// is needed (symbols resolve through the dlopen wrapper). Whether the opened
// device is ionic comes from ionic_dv_is_ionic_ctx when libionic loads (NIC names
// such as rdma0 carry no vendor), else from the sysfs NIC family.

#ifndef ACCEPT_HAVE_IBVERBS
// No ibverbs -- nothing to register.
#else

#include "verbs/PostCost.hpp"
#include "verbs/TwoNode.hpp"  // rc_loopback_to_rts
#include "wrapper/verbs/IbvApi.hpp"
#include "wrapper/verbs/IonicDvApi.hpp"
#include "wrapper/verbs/VerbsHelpers.hpp"

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"
#include "common/Topology.hpp"

namespace acceptance {

namespace {

namespace pc = postcost;

constexpr int kDepth = 128;           // max_send_wr; matches postlat.c / RCCL
constexpr int kBatch = 8;             // every kBatch-th WR is signalled
constexpr int kBlock = 64;            // posts per timed block (see run_config)
constexpr int kWarmBlocks = 32;       // unrecorded blocks per config
constexpr int kBlocks = 512;          // recorded blocks per config (~32k posts)
constexpr int kLatWarmup = 500;       // perf mode: post->CQE warmup
constexpr int kLatSamples = 3000;     // perf mode: post->CQE samples
constexpr uint64_t kLatWrId = ~0ull;  // marks the latency WR among throughput CQEs
constexpr uint64_t kMarkWrId = ~0ull - 1;  // end-of-run flush marker (one per QP)
constexpr std::chrono::seconds kStallBudget{10};

// One WQE shape: how many QPs are driven round-robin, the RDMA_WRITE size, and
// whether the payload rides inline in the WQE.
struct Shape {
  const char* name;
  int         qps;
  uint32_t    bytes;
  bool        inl;
};

constexpr Shape kShapes[] = {
    {"qp1.inl64", 1, 64, true},   {"qp16.inl64", 16, 64, true},
    {"qp64.inl64", 64, 64, true}, {"qp16.inl8", 16, 8, true},
    {"qp16.sge1024", 16, 1024, false},
};

// One SQ placement. `sq_cmb_env` is the IONIC_SQ_CMB value set while the context
// opens (nullptr = unset, i.e. libionic defaults), `pd_cmb_off` also calls
// ionic_dv_pd_set_sqcmb(pd, false, ...) on the PD.
struct Variant {
  const char* label;
  const char* sq_cmb_env;
  bool        pd_cmb_off;
  bool        require_cmb;   // skip the variant unless the SQ lands in the CMB
  bool        require_host;  // skip the variant if the SQ lands in the CMB
  const char* knob;          // how to select it, for the recommendation
};

constexpr Variant kDefault = {"default", nullptr, false, false, false,
                              "libionic out-of-box default"};
constexpr Variant kAlternatives[] = {
    {"host", "0", false, false, true,
     "IONIC_SQ_CMB=0 (RCCL: RCCL_AINIC_SQ_CMB=0)"},
    {"host_pd", nullptr, true, false, true,
     "ionic_dv_pd_set_sqcmb(pd, false, ...) (RCCL: RCCL_AINIC_SQ_CMB=0)"},
    {"cmb_ew", "ew", false, true, false, "IONIC_SQ_CMB=ew (CMB, write-combining)"},
    {"cmb_xw", "xw", false, true, false,
     "IONIC_SQ_CMB=xw (CMB + express doorbell, write-combining)"},
};

// Set/unset an environment variable for a scope and restore the caller's value,
// so a scenario never leaks into the next test of an --isolate none run.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_ = true;
      old_ = old;
    }
    apply(value);
  }
  ~ScopedEnv() { apply(had_ ? old_.c_str() : nullptr); }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  void apply(const char* value) {
    if (value != nullptr) ::setenv(name_, value, 1);
    else ::unsetenv(name_);
  }
  const char* name_;
  bool        had_ = false;
  std::string old_;
};

long uverbs_mapped_kb() {
  std::ifstream maps("/proc/self/maps");
  return pc::sum_mapped_kb(maps, "uverbs");
}

int roce_gid_index(const std::string& dev) {
  for (const RdmaDevice& d : environment().rdma_devices)
    if (d.device == dev && d.gid_index >= 0) return d.gid_index;
  return -1;
}

// Device context + PD + CQ + MR shared by every config of one test. Teardown is
// noexcept (safe_* helpers) so an early return can never throw out of it.
class Rig {
 public:
  ~Rig() {
    ibvw::safe_dereg_mr(mr_);
    ibvw::safe_destroy_cq(cq_);
    ibvw::safe_dealloc_pd(pd_);
    ibvw::safe_close_device(ctx_);
    ibvw::safe_free_device_list(list_);
  }

  // Open the selected HCA under the scenario's environment and set up the PD
  // (with the PD-level SQ call when asked). Returns "" or a reason; `no_dev`
  // marks the "no RDMA hardware" skip; `ionic_hw` says whether the opened device
  // is an ionic one (libionic's answer when it is loadable, else the NIC family
  // detected from sysfs -- anything that is not a known other vendor counts).
  std::string open(const Variant& sc, bool& no_dev, bool& ionic_hw) {
    no_dev = false;
    ionic_hw = false;
    int num = 0;
    list_ = ibv_get_device_list(&num);
    if (list_ == nullptr) return ibvw::errno_reason("ibv_get_device_list failed");
    if (num == 0) {
      no_dev = true;
      return "no RDMA devices present on this host";
    }
    int idx = 0;
    if (const std::string want = topology::get_rank_hca(); !want.empty()) {
      idx = -1;
      for (int i = 0; i < num; ++i) {
        const char* n = ibv_get_device_name(list_[i]);
        if (n != nullptr && want == n) { idx = i; break; }
      }
      if (idx < 0) return "IB device '" + want + "' not found";
    }
    {
      // libionic reads IONIC_SQ_CMB while it allocates the context.
      ScopedEnv env("IONIC_SQ_CMB", sc.sq_cmb_env);
      ctx_ = ibv_open_device(list_[idx]);
    }
    if (ctx_ == nullptr) return ibvw::errno_reason("ibv_open_device failed");
    const char* n = ibv_get_device_name(list_[idx]);
    dev_ = n ? n : "<unknown>";
    if (auto is_ionic = ionicdv::is_ionic_ctx(); is_ionic != nullptr) {
      ionic_hw = is_ionic(ctx_);
    } else {
      const std::string& vendor = environment().vendor;
      ionic_hw = vendor != "thor" && vendor != "connectx";
    }

    sgid_ = roce_gid_index(dev_);
    if (sgid_ < 0) {
      no_dev = true;  // skip-worthy: nothing routable to loop back through
      return "no routable RoCEv2 GID index for " + dev_;
    }
    std::memset(&gid_, 0, sizeof(gid_));
    if (ibv_query_gid(ctx_, 1, sgid_, &gid_) != 0)
      return "ibv_query_gid failed: " + std::string(std::strerror(errno));

    pd_ = ibv_alloc_pd(ctx_);
    if (pd_ == nullptr) return ibvw::errno_reason("ibv_alloc_pd failed");
    if (sc.pd_cmb_off) {
      // Same call RCCL makes for RCCL_AINIC_SQ_CMB=0: enable=false, expdb=true.
      const int rc = ionicdv::pd_set_sqcmb()(pd_, false, true, false);
      if (rc != 0)
        return "ionic_dv_pd_set_sqcmb(false) rc=" + std::to_string(rc) + ": " +
               std::strerror(std::abs(rc));
    }
    cq_ = ibv_create_cq(ctx_, 4096, nullptr, nullptr, 0);
    if (cq_ == nullptr) return ibvw::errno_reason("ibv_create_cq failed");
    buf_.assign(1 << 20, 0x5a);
    mr_ = ibv_reg_mr(pd_, buf_.data(), buf_.size(),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (mr_ == nullptr) return ibvw::errno_reason("ibv_reg_mr failed");
    return "";
  }

  struct ibv_pd*      pd() const { return pd_; }
  struct ibv_cq*      cq() const { return cq_; }
  struct ibv_mr*      mr() const { return mr_; }
  uint8_t*            buf() { return buf_.data(); }
  const union ibv_gid& gid() const { return gid_; }
  int                 sgid() const { return sgid_; }
  const std::string&  dev() const { return dev_; }

 private:
  struct ibv_device** list_ = nullptr;
  struct ibv_context* ctx_ = nullptr;
  struct ibv_pd*      pd_ = nullptr;
  struct ibv_cq*      cq_ = nullptr;
  struct ibv_mr*      mr_ = nullptr;
  std::vector<uint8_t> buf_;
  union ibv_gid       gid_;
  int                 sgid_ = -1;
  std::string         dev_;
};

struct ConfigResult {
  std::string         error;          // hard setup/data-path failure
  std::string         skip;           // config cannot run here (not a defect)
  pc::Stats           post;           // ns
  double              kb_per_qp = 0;  // uverbs KiB mapped per QP (CMB SQ > 0)
  std::vector<double> post_us;        // for perf metrics
  std::vector<double> lat_us;         // post->CQE (perf mode only)
};

// What reap() saw besides ordinary throughput completions.
struct Seen {
  bool lat = false;  // the post->CQE latency WR completed
  int  marks = 0;    // end-of-run flush markers completed
};

// Reap up to 64 CQEs, retiring kBatch WRs per throughput completion. Returns ""
// or a reason on a bad completion / poll error.
std::string reap(struct ibv_cq* cq, std::vector<int>& pend, Seen& seen) {
  struct ibv_wc wc[64];
  const int n = ibv_poll_cq(cq, 64, wc);
  if (n < 0) return "ibv_poll_cq failed";
  for (int k = 0; k < n; ++k) {
    if (wc[k].status != IBV_WC_SUCCESS)
      return "completion error status=" + std::to_string(wc[k].status) +
             " vendor_err=" + std::to_string(wc[k].vendor_err);
    if (wc[k].wr_id == kLatWrId) {
      seen.lat = true;
    } else if (wc[k].wr_id == kMarkWrId) {
      ++seen.marks;
    } else if (wc[k].wr_id < pend.size()) {
      pend[wc[k].wr_id] -= kBatch;
    }
  }
  return "";
}

ConfigResult run_config(Rig& rig, const Shape& shape, bool want_lat) {
  ConfigResult r;
  const int nq = shape.qps;

  // Create the QPs; the uverbs mapping delta per QP tells where the SQ lives.
  struct Qps {
    std::vector<struct ibv_qp*> qp;
    ~Qps() { for (auto* q : qp) ibvw::safe_destroy_qp(q); }
  } qps;
  const long kb0 = uverbs_mapped_kb();
  uint32_t granted_inline = 0;
  for (int i = 0; i < nq; ++i) {
    struct ibv_qp_init_attr qia;
    std::memset(&qia, 0, sizeof(qia));
    qia.send_cq = rig.cq();
    qia.recv_cq = rig.cq();
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = kDepth;
    qia.cap.max_recv_wr = 16;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 128;
    struct ibv_qp* qp = ibv_create_qp(rig.pd(), &qia);
    if (qp == nullptr) {
      r.error = std::string(shape.name) + ": ibv_create_qp #" +
                std::to_string(i) + " failed: " + std::strerror(errno);
      return r;
    }
    qps.qp.push_back(qp);
    granted_inline = qia.cap.max_inline_data;
  }
  r.kb_per_qp = static_cast<double>(uverbs_mapped_kb() - kb0) / nq;

  if (shape.inl && granted_inline < shape.bytes) {
    r.skip = std::string(shape.name) + ": provider granted max_inline_data=" +
             std::to_string(granted_inline) + " < " + std::to_string(shape.bytes);
    return r;
  }
  for (struct ibv_qp* qp : qps.qp) {
    if (std::string e = verbsx::rc_loopback_to_rts(qp, rig.gid(), rig.sgid());
        !e.empty()) {
      r.error = std::string(shape.name) + ": QP bring-up: " + e;
      return r;
    }
  }

  struct ibv_sge sge;
  sge.addr = reinterpret_cast<uintptr_t>(rig.buf());
  sge.length = shape.bytes;
  sge.lkey = rig.mr()->lkey;

  std::vector<int> pend(nq, 0), cnt(nq, 0);
  Seen seen;
  std::vector<double> post_ns;  // per-call cost, one entry per timed block
  post_ns.reserve(kBlocks);
  const auto deadline = PerfClock::now() + kStallBudget;

  // Each QP receives at most `per_qp` posts per block; it needs that much room
  // below the wrap point (depth minus one signalling batch) before the block.
  const int per_qp = (kBlock + nq - 1) / nq;
  std::vector<struct ibv_send_wr> wrs(kBlock);
  std::vector<int> wq(kBlock);

  for (int b = 0; b < kWarmBlocks + kBlocks; ++b) {
    for (int q = 0; q < nq; ++q) {
      while (pend[q] + per_qp > kDepth - kBatch) {  // keep the SQ from wrapping
        if (std::string e = reap(rig.cq(), pend, seen); !e.empty()) {
          r.error = std::string(shape.name) + ": " + e;
          return r;
        }
        if (PerfClock::now() > deadline) {
          r.error = std::string(shape.name) + ": send completions stalled";
          return r;
        }
      }
    }

    // Build the block's WRs outside the timed window.
    for (int j = 0; j < kBlock; ++j) {
      const int q = (b * kBlock + j) % nq;
      wq[j] = q;
      struct ibv_send_wr& wr = wrs[j];
      std::memset(&wr, 0, sizeof(wr));
      wr.wr_id = static_cast<uint64_t>(q);
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.send_flags = (shape.inl ? IBV_SEND_INLINE : 0) |
                      ((cnt[q]++ % kBatch == kBatch - 1) ? IBV_SEND_SIGNALED : 0);
      wr.wr.rdma.remote_addr =
          reinterpret_cast<uintptr_t>(rig.buf()) + 4096 + (q % 64) * 64;
      wr.wr.rdma.rkey = rig.mr()->rkey;
    }

    int rc = 0;
    int posted = 0;
    const auto t0 = PerfClock::now();
    for (; posted < kBlock; ++posted) {
      struct ibv_send_wr* bad = nullptr;
      rc = ibv_post_send(qps.qp[wq[posted]], &wrs[posted], &bad);
      if (rc != 0) break;
    }
    const auto t1 = PerfClock::now();
    if (rc != 0) {
      r.error = std::string(shape.name) + ": ibv_post_send failed: " +
                std::strerror(rc > 0 ? rc : errno);
      return r;
    }
    for (int j = 0; j < kBlock; ++j) ++pend[wq[j]];
    if (b >= kWarmBlocks)
      post_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                        kBlock);
    if (std::string e = reap(rig.cq(), pend, seen); !e.empty()) {
      r.error = std::string(shape.name) + ": " + e;
      return r;
    }
  }

  // Drain. A signalled marker per QP completes behind every WR posted before
  // it, so once all markers are back nothing is in flight; destroying a QP with
  // WRs still outstanding would leave flush CQEs that libionic then logs as
  // "missing qp" on the next poll of the shared CQ.
  for (int q = 0; q < nq; ++q) {
    struct ibv_send_wr wr, *bad = nullptr;
    std::memset(&wr, 0, sizeof(wr));
    wr.wr_id = kMarkWrId;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED | (shape.inl ? IBV_SEND_INLINE : 0);
    wr.wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(rig.buf()) + 4096;
    wr.wr.rdma.rkey = rig.mr()->rkey;
    for (;;) {  // the SQ may be momentarily full of earlier WRs
      const int rc = ibv_post_send(qps.qp[q], &wr, &bad);
      if (rc == 0) break;
      if (rc != ENOMEM && rc != EAGAIN) {
        r.error = std::string(shape.name) + ": drain ibv_post_send failed: " +
                  std::strerror(rc > 0 ? rc : errno);
        return r;
      }
      if (std::string e = reap(rig.cq(), pend, seen); !e.empty()) {
        r.error = std::string(shape.name) + ": " + e;
        return r;
      }
      if (PerfClock::now() > deadline) {
        r.error = std::string(shape.name) + ": drain stalled";
        return r;
      }
    }
  }
  while (seen.marks < nq) {
    if (std::string e = reap(rig.cq(), pend, seen); !e.empty()) {
      r.error = std::string(shape.name) + ": " + e;
      return r;
    }
    if (PerfClock::now() > deadline) {
      r.error = std::string(shape.name) + ": drain stalled";
      return r;
    }
  }

  r.post = pc::summarize(post_ns);
  for (double ns : post_ns) r.post_us.push_back(ns / 1000.0);

  // post -> CQE of one signalled WR on QP0 (perf mode only; never gates).
  if (want_lat) {
    for (int i = 0; i < kLatWarmup + kLatSamples; ++i) {
      struct ibv_send_wr wr, *bad = nullptr;
      std::memset(&wr, 0, sizeof(wr));
      wr.wr_id = kLatWrId;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.send_flags = IBV_SEND_SIGNALED | (shape.inl ? IBV_SEND_INLINE : 0);
      wr.wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(rig.buf()) + 4096;
      wr.wr.rdma.rkey = rig.mr()->rkey;
      const auto t0 = PerfClock::now();
      if (ibv_post_send(qps.qp[0], &wr, &bad) != 0) break;
      seen.lat = false;
      const auto lat_deadline = PerfClock::now() + kStallBudget;
      while (!seen.lat) {
        if (!reap(rig.cq(), pend, seen).empty() ||
            PerfClock::now() > lat_deadline)
          break;
      }
      if (!seen.lat) break;
      if (i >= kLatWarmup)
        r.lat_us.push_back(
            std::chrono::duration<double, std::micro>(PerfClock::now() - t0).count());
    }
  }
  return r;
}

// All shapes of one placement.
struct VariantResult {
  std::string               error;     // hard failure
  std::string               skip;      // placement not applicable here
  bool                      no_dev = false;
  bool                      ionic_hw = false;
  std::vector<pc::Stats>    post;      // per kShapes entry; n==0 = shape skipped
  std::vector<std::string>  notes;     // shapes that could not run
  double                    kb_per_qp = 0;
};

constexpr std::size_t kNumShapes = sizeof(kShapes) / sizeof(kShapes[0]);

VariantResult measure_variant(const Variant& v, PerfSink& sink) {
  VariantResult out;
  out.post.assign(kNumShapes, pc::Stats{});

  Rig rig;
  if (std::string e = rig.open(v, out.no_dev, out.ionic_hw); !e.empty()) {
    out.error = e;
    return out;
  }

  bool first = true;
  for (std::size_t i = 0; i < kNumShapes; ++i) {
    const Shape& shape = kShapes[i];
    ConfigResult r = run_config(rig, shape, perf_probe_mode());
    if (!r.error.empty()) {
      out.error = std::string(v.label) + ": " + r.error;
      return out;
    }
    if (!r.skip.empty()) {
      out.notes.push_back(r.skip);
      continue;
    }

    // Placement is a property of the variant: judge it on the first shape that
    // ran. A CMB SQ adds one queue-sized uverbs mapping per QP.
    const bool in_cmb = r.kb_per_qp >= pc::kCmbMapKbPerQp;
    if (first) {
      first = false;
      out.kb_per_qp = r.kb_per_qp;
      if (!out.ionic_hw) {
        // No SQ placement concept: only the default is ever measured here.
      } else if (v.require_cmb && !in_cmb) {
        out.skip = "SQ did not land in the CMB (no per-QP uverbs mapping)";
        return out;
      }
      if (v.require_host && in_cmb) {
        out.skip = "SQ landed in the CMB (" +
                   std::to_string(static_cast<long>(r.kb_per_qp)) +
                   " KiB mapped per QP) although host memory was requested";
        return out;
      }
    }
    out.post[i] = r.post;
    emit_latency_samples(sink, std::string("post.") + v.label + "." + shape.name,
                         r.post_us);
    emit_latency_samples(sink, std::string("lat.") + v.label + "." + shape.name,
                         r.lat_us);
  }
  if (first) out.skip = "no WQE shape could run";
  return out;
}

std::string ns(double v) { return perf_detail::format_fixed(v, 0); }

// Healthy-median key for the hardcoded reference: "ionic", or the NIC family.
std::string reference_key(bool ionic_hw) {
  if (ionic_hw) return "ionic";
  const std::string& vendor = environment().vendor;
  return vendor.empty() ? "unknown" : vendor;
}

TestResult run_test() {
  // Optional helpers: loads libionic when present; the test must not need them.
  const bool have_libionic = ionicdv::init();

  const double tol = pc::tolerance_frac();
  PerfSink sink;

  VariantResult def = measure_variant(kDefault, sink);
  if (!def.error.empty()) return def.no_dev ? skip(def.error) : fail(def.error);
  if (!def.skip.empty()) return skip("default: " + def.skip);
  const std::string key = reference_key(def.ionic_hw);

  std::string tag = "[post-cost] " + key + " NIC, default SQ " +
                    std::string(!def.ionic_hw ? "(no SQ placement choice)"
                                : def.kb_per_qp >= pc::kCmbMapKbPerQp
                                    ? "in CMB (" +
                                          std::to_string(static_cast<long>(def.kb_per_qp)) +
                                          " KiB mapped/QP)"
                                    : "in host memory") +
                    ", median ns/call (95% CI), tolerance " +
                    std::to_string(static_cast<int>(tol * 100 + 0.5)) + "%";
  std::fprintf(stderr, "%s\n", tag.c_str());

  // ---- measure the alternatives when libionic lets us select them -----------
  struct Alt {
    const Variant*   v;
    VariantResult    res;
  };
  std::vector<Alt> alts;
  if (have_libionic && def.ionic_hw) {
    for (const Variant& v : kAlternatives) {
      if (v.pd_cmb_off && ionicdv::pd_set_sqcmb() == nullptr) {
        std::fprintf(stderr, "[post-cost]   %-8s skipped: ionic_dv_pd_set_sqcmb not exported\n",
                     v.label);
        continue;
      }
      VariantResult r = measure_variant(v, sink);
      if (!r.error.empty() || !r.skip.empty()) {
        std::fprintf(stderr, "[post-cost]   %-8s skipped: %s\n", v.label,
                     (r.error.empty() ? r.skip : r.error).c_str());
        continue;
      }
      alts.push_back({&v, std::move(r)});
    }
  }

  // ---- judge the default -----------------------------------------------------
  std::string broken;  // empty = default is fine
  if (!alts.empty()) {
    // Worst shape = largest default/alternative ratio over flagged pairs.
    double worst_ratio = 0;
    std::size_t worst_shape = 0;
    const Alt* worst_alt = nullptr;
    int flagged = 0, judged = 0;
    for (std::size_t i = 0; i < kNumShapes; ++i) {
      if (def.post[i].n == 0) continue;
      ++judged;
      std::string line = std::string("[post-cost]   ") + kShapes[i].name +
                         ": default " + ns(def.post[i].median) + " (" +
                         ns(def.post[i].ci_lo) + "-" + ns(def.post[i].ci_hi) + ")";
      bool shape_flagged = false;
      for (const Alt& a : alts) {
        const pc::Stats& st = a.res.post[i];
        if (st.n == 0) continue;
        const double ratio = def.post[i].median / st.median;
        line += " | " + std::string(a.v->label) + " " + ns(st.median);
        if (pc::clearly_slower(def.post[i], st, tol)) {
          line += " [default " + perf_detail::format_fixed(ratio, 1) + "x slower]";
          shape_flagged = true;
          if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_shape = i;
            worst_alt = &a;
          }
        } else if (pc::clearly_slower(st, def.post[i], tol)) {
          line += " [" + std::string(a.v->label) + " " +
                  perf_detail::format_fixed(st.median / def.post[i].median, 1) +
                  "x slower than default]";
        }
      }
      std::fprintf(stderr, "%s\n", line.c_str());
      if (shape_flagged) ++flagged;
    }
    if (flagged > 0) {
      // Recommend the alternative that is fastest at the worst shape.
      const Alt* best = worst_alt;
      for (const Alt& a : alts)
        if (a.res.post[worst_shape].n != 0 &&
            a.res.post[worst_shape].median < best->res.post[worst_shape].median)
          best = &a;
      const pc::Stats& d = def.post[worst_shape];
      const pc::Stats& b = best->res.post[worst_shape];
      broken = "default (out-of-box) ibv_post_send is " + ns(d.median) +
               " ns/call (95% CI " + ns(d.ci_lo) + "-" + ns(d.ci_hi) + ") on " +
               kShapes[worst_shape].name + ", " +
               perf_detail::format_fixed(d.median / b.median, 1) + "x slower than " +
               best->v->label + " (" + ns(b.median) + " ns), over the " +
               std::to_string(static_cast<int>(tol * 100 + 0.5)) +
               "% tolerance in " + std::to_string(flagged) + "/" +
               std::to_string(judged) + " configs; default SQ " +
               (def.kb_per_qp >= pc::kCmbMapKbPerQp
                    ? "is in the CMB (" +
                          std::to_string(static_cast<long>(def.kb_per_qp)) +
                          " KiB uverbs mapping/QP), likely mapped uncached so every "
                          "WQE byte is an MMIO store"
                    : "is in host memory") +
               ". Recommendation: select " + best->v->knob;
    }
  } else {
    // No alternatives measurable: compare with the hardcoded healthy figure.
    const double ref = pc::reference_ns_for(key);
    int flagged = 0, judged = 0;
    std::size_t worst_shape = 0;
    double worst_lo = 0;
    for (std::size_t i = 0; i < kNumShapes; ++i) {
      if (def.post[i].n == 0) continue;
      ++judged;
      const bool over = pc::above_reference(def.post[i], ref, tol);
      std::fprintf(stderr, "[post-cost]   %s: default %s (%s-%s) vs reference %s%s\n",
                   kShapes[i].name, ns(def.post[i].median).c_str(),
                   ns(def.post[i].ci_lo).c_str(), ns(def.post[i].ci_hi).c_str(),
                   ns(ref).c_str(), over ? " [above reference]" : "");
      if (over) {
        ++flagged;
        if (def.post[i].ci_lo > worst_lo) {
          worst_lo = def.post[i].ci_lo;
          worst_shape = i;
        }
      }
    }
    if (flagged > 0) {
      const pc::Stats& d = def.post[worst_shape];
      broken = "default (out-of-box) ibv_post_send is " + ns(d.median) +
               " ns/call (95% CI " + ns(d.ci_lo) + "-" + ns(d.ci_hi) + ") on " +
               kShapes[worst_shape].name + ", above the " + ns(ref) + " ns expected " +
               key + " figure + " + std::to_string(static_cast<int>(tol * 100 + 0.5)) +
               "% tolerance in " + std::to_string(flagged) + "/" +
               std::to_string(judged) + " configs";
      if (!def.ionic_hw) {
        broken += "; check the NIC driver/firmware and that the doorbell/WQE "
                  "mapping is write-combining";
      } else {
        broken += std::string(" (") +
                  (have_libionic ? "no alternative SQ placement could be measured"
                                 : "libionic direct-verbs unavailable, so alternative "
                                   "SQ placements were not measured") +
                  "); default SQ " +
                  (def.kb_per_qp >= pc::kCmbMapKbPerQp
                       ? "is in the CMB (" +
                             std::to_string(static_cast<long>(def.kb_per_qp)) +
                             " KiB uverbs mapping/QP), likely mapped uncached"
                       : "is in host memory") +
                  ". Recommendation: try IONIC_SQ_CMB=0 (RCCL: RCCL_AINIC_SQ_CMB=0) "
                  "or a write-combining CMB mapping (IONIC_SQ_CMB=xw) and re-run";
      }
    }
  }

  if (!broken.empty()) {
    TestResult res = fail(broken);
    res.perf = std::move(sink);
    return res;
  }
  return pass_with_perf(std::move(sink));
}

}  // namespace

ACCEPT_TEST(vat_post_send_cost, "verbs/datapath", Topology::Single,
            Kind::Functional, 180) {
  (void)ctx;
  return run_test();
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
