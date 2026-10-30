/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/RcConn.hpp"

#include <arpa/inet.h>  // htonl
#include <time.h>       // nanosleep

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "common/Capability.hpp"
#include "common/Topology.hpp"

namespace acceptance {
namespace verbsx {

namespace {

// CQ depth: this endpoint only ever has a handful of WRs in flight at once
// (the tests post one signaled op and poll it), so a small fixed depth is ample.
constexpr int kCqDepth = 32;

// Resolve the routable RoCEv2 GID index for `dev` from the shared capability
// probe (Capability reads sysfs gid_attrs/types and picks the "RoCE v2" IPv4
// entry -- index 3 on these mlx5 parts, not the hardcoded 1).
// Returns -1 when the device is not found / no RoCEv2 GID present.
int roce_v2_gid_index(const std::string& dev) {
  for (const auto& d : environment().rdma_devices)
    if (d.device == dev && d.gid_index >= 0) return d.gid_index;
  return -1;
}

// Sleep a small fixed interval between CQ polls (matches the reference cadence).
void sleep_us(long us) {
  struct timespec ts;
  ts.tv_sec = us / 1000000L;
  ts.tv_nsec = (us % 1000000L) * 1000L;
  ::nanosleep(&ts, nullptr);
}

std::string errno_str(const char* what) { return ibvw::errno_reason(what); }

}  // namespace

RcConn::~RcConn() {
  // noexcept teardown through the null-guarded ops-table pointers (never the
  // throwing ibv_* macros) so a failure here cannot std::terminate during stack
  // unwinding from a MissingSymbol.
  ibvw::safe_destroy_qp(qp_);
  ibvw::safe_destroy_cq(cq_);
  ibvw::safe_dealloc_pd(pd_);
  ibvw::safe_close_device(ctx_);
}

std::string RcConn::open(const std::string& dev_name, bool& no_device,
                         uint32_t max_inline) {
  no_device = false;
  // Empty dev_name means "auto-select": open this rank's selected HCA
  // (ACCEPT_HCA, see Topology.hpp) -- the device the environment report and
  // its GID lookup name -- so the connect path uses that NIC too.
  std::string target = dev_name.empty() ? topology::get_rank_hca() : dev_name;
  int num = 0;
  struct ibv_device** list = ibv_get_device_list(&num);
  if (list == nullptr) return errno_str("ibv_get_device_list failed");
  if (num == 0) {
    ibvw::safe_free_device_list(list);
    no_device = true;
    return "no RDMA devices present on this host";
  }

  struct ibv_device* dev = nullptr;
  if (!target.empty()) {
    for (int i = 0; i < num; ++i) {
      const char* name = ibv_get_device_name(list[i]);
      if (name != nullptr && target == name) { dev = list[i]; break; }
    }
    if (dev == nullptr) {
      ibvw::safe_free_device_list(list);
      return "IB device '" + target + "' not found";
    }
    ctx_ = ibv_open_device(dev);
  } else {
    // First device with an ACTIVE port.
    for (int i = 0; i < num && ctx_ == nullptr; ++i) {
      struct ibv_context* candidate = ibv_open_device(list[i]);
      if (candidate == nullptr) continue;
      struct ibv_port_attr pa;
      std::memset(&pa, 0, sizeof(pa));
      if (ibv_query_port(candidate, ib_port_, &pa) == 0 &&
          pa.state == IBV_PORT_ACTIVE) {
        dev = list[i];
        ctx_ = candidate;
      } else {
        ibvw::safe_close_device(candidate);
      }
    }
    if (ctx_ == nullptr) {
      ibvw::safe_free_device_list(list);
      no_device = true;  // no linked-up port anywhere -> skip-worthy
      return "no RDMA device with an ACTIVE port";
    }
  }

  if (ctx_ == nullptr) {
    ibvw::safe_free_device_list(list);
    return errno_str("ibv_open_device failed");
  }
  const char* dev_name_cstr = ibv_get_device_name(dev);
  dev_name_ = dev_name_cstr ? dev_name_cstr : "<unknown>";
  ibvw::safe_free_device_list(list);

  // Pick the routable RoCEv2 GID index for this device.
  gid_index_ = roce_v2_gid_index(dev_name_);
  // gid_index_ == -1 means no routable RoCEv2 GID was found in sysfs.
  // Tests that need a routable GID (vat_ah_lifecycle, cross-node connects)
  // check gid_index() < 0 and skip cleanly.

  struct ibv_port_attr pa;
  std::memset(&pa, 0, sizeof(pa));
  if (ibv_query_port(ctx_, ib_port_, &pa) != 0)
    return errno_str("ibv_query_port failed");
  local_.lid = pa.lid;
  // Use the port's negotiated MTU, not a fixed guess: requesting a path_mtu the
  // port never advertised (e.g. 4096 on a port active at 1024) fails RTR with
  // EINVAL. RoCE has no per-path MTU discovery, so the local active_mtu is the
  // right bound (matches on both ends after the deterministic RESET->INIT path).
  path_mtu_ = pa.active_mtu;

  if (gid_index_ >= 0) {
    union ibv_gid gid;
    std::memset(&gid, 0, sizeof(gid));
    if (ibv_query_gid(ctx_, ib_port_, gid_index_, &gid) != 0)
      return errno_str("ibv_query_gid failed");
    std::memcpy(local_.gid, gid.raw, 16);
  }

  pd_ = ibv_alloc_pd(ctx_);
  if (pd_ == nullptr) return errno_str("ibv_alloc_pd failed");

  cq_ = ibv_create_cq(ctx_, kCqDepth, nullptr, nullptr, 0);
  if (cq_ == nullptr) return errno_str("ibv_create_cq failed");

  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq_;
  qia.recv_cq = cq_;
  qia.qp_type = IBV_QPT_RC;
  qia.sq_sig_all = 0;
  qia.cap.max_send_wr = 16;
  qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1;
  qia.cap.max_recv_sge = 1;
  // The default max_inline_data=0 matches RCCL's default (NCCL_IB_USE_INLINE=0).
  // Providers differ in what they grant: ionic rounds 0 up to 24 (minimum WQE
  // size), bnxt_re grants exactly what was asked. If a non-zero request is
  // rejected, retry with 0 so the caller can skip on max_inline() instead.
  // Intentional: the retry differs only in max_inline_data, so any other
  // create fault fails it too and still surfaces as "ibv_create_qp failed".
  qia.cap.max_inline_data = max_inline;
  qp_ = ibv_create_qp(pd_, &qia);
  if (qp_ == nullptr && max_inline > 0) {
    qia.cap.max_inline_data = 0;
    qp_ = ibv_create_qp(pd_, &qia);
  }
  if (qp_ == nullptr) return errno_str("ibv_create_qp failed");
  // ibv_create_qp(3) writes the granted caps back into qia, but prefer asking
  // the QP itself in case a provider only reports them via ibv_query_qp. A
  // failed query keeps the create-time value rather than failing every RC test.
  max_inline_ = qia.cap.max_inline_data;
  struct ibv_qp_attr cap_attr;
  struct ibv_qp_init_attr cap_init;
  std::memset(&cap_attr, 0, sizeof(cap_attr));
  std::memset(&cap_init, 0, sizeof(cap_init));
  if (ibv_query_qp(qp_, &cap_attr, IBV_QP_CAP, &cap_init) == 0)
    max_inline_ = cap_attr.cap.max_inline_data;

  // Query device caps to gate optional features: don't request REMOTE_ATOMIC
  // on a NIC that advertises no atomic support (some drivers return EINVAL).
  struct ibv_device_attr dev_attr;
  std::memset(&dev_attr, 0, sizeof(dev_attr));
  has_atomics_ = (ibv_query_device(ctx_, &dev_attr) == 0 &&
                  dev_attr.atomic_cap != IBV_ATOMIC_NONE);

  // RESET -> INIT. Allow local + remote WRITE/READ, plus atomics when supported.
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_INIT;
  qa.pkey_index = 0;
  qa.port_num = static_cast<uint8_t>(ib_port_);
  qa.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ |
                       (has_atomics_ ? IBV_ACCESS_REMOTE_ATOMIC : 0);
  if (ibv_modify_qp(qp_, &qa,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) != 0)
    return errno_str("ibv_modify_qp INIT failed");

  local_.qpn = qp_->qp_num;
  local_.psn = 0;  // deterministic; loopback needs rq_psn == sq_psn
  return "";
}

std::string RcConn::to_rts(const QpInfo& remote) {
  // INIT -> RTR (RoCE: is_global, GID-addressed, LID unused).
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_RTR;
  qa.path_mtu = path_mtu_;
  qa.dest_qp_num = remote.qpn;
  qa.rq_psn = remote.psn;
  qa.max_dest_rd_atomic = 16;
  qa.min_rnr_timer = 12;
  qa.ah_attr.is_global = 1;
  qa.ah_attr.dlid = remote.lid;
  qa.ah_attr.sl = 0;
  qa.ah_attr.src_path_bits = 0;
  qa.ah_attr.port_num = static_cast<uint8_t>(ib_port_);
  std::memcpy(qa.ah_attr.grh.dgid.raw, remote.gid, 16);
  qa.ah_attr.grh.sgid_index = static_cast<uint8_t>(gid_index_);
  qa.ah_attr.grh.hop_limit = 64;
  if (ibv_modify_qp(qp_, &qa,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
    return errno_str("ibv_modify_qp RTR failed");

  // RTR -> RTS.
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_RTS;
  qa.sq_psn = local_.psn;
  qa.timeout = 14;
  qa.retry_cnt = 7;
  qa.rnr_retry = 7;
  qa.max_rd_atomic = 16;
  if (ibv_modify_qp(qp_, &qa,
                    IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                        IBV_QP_MAX_QP_RD_ATOMIC) != 0)
    return errno_str("ibv_modify_qp RTS failed");
  return "";
}

std::string RcConn::to_err() {
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_ERR;
  if (ibv_modify_qp(qp_, &qa, IBV_QP_STATE) != 0)
    return errno_str("ibv_modify_qp ERR failed");
  return "";
}

std::string RcConn::to_reset() {
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_RESET;
  if (ibv_modify_qp(qp_, &qa, IBV_QP_STATE) != 0)
    return errno_str("ibv_modify_qp RESET failed");
  return "";
}

std::string RcConn::reinit() {
  // RESET -> INIT, re-applying the same access flags/port as open(). The QP
  // handle is reused; the caller then re-runs to_rts() to bring it back up.
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_INIT;
  qa.pkey_index = 0;
  qa.port_num = static_cast<uint8_t>(ib_port_);
  qa.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ |
                       (has_atomics_ ? IBV_ACCESS_REMOTE_ATOMIC : 0);
  if (ibv_modify_qp(qp_, &qa,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) != 0)
    return errno_str("ibv_modify_qp reINIT failed");
  local_.qpn = qp_->qp_num;
  local_.psn = 0;
  return "";
}

std::string RcConn::post_send_one(struct ibv_send_wr& wr) {
  struct ibv_send_wr* bad = nullptr;
  // ibv_post_send is a static inline dispatching through qp->context->ops; it is
  // NOT a wrapper macro, so this is a direct provider call.
  if (ibv_post_send(qp_, &wr, &bad) != 0)
    return errno_str("ibv_post_send failed");
  return "";
}

std::string RcConn::post_rdma(ibv_wr_opcode opcode, uint32_t lkey,
                              void* local_addr, uint32_t len,
                              const RemoteBuf& rb, uint64_t wr_id, uint32_t imm) {
  struct ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uint64_t>(local_addr);
  sge.length = len;
  sge.lkey = lkey;

  struct ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.opcode = opcode;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (opcode == IBV_WR_RDMA_WRITE_WITH_IMM) wr.imm_data = htonl(imm);
  wr.wr.rdma.remote_addr = rb.addr;
  wr.wr.rdma.rkey = rb.rkey;
  return post_send_one(wr);
}

std::string RcConn::post_write(uint32_t lkey, void* local_addr, uint32_t len,
                               const RemoteBuf& rb, uint64_t wr_id) {
  return post_rdma(IBV_WR_RDMA_WRITE, lkey, local_addr, len, rb, wr_id);
}

std::string RcConn::post_write_imm(uint32_t lkey, void* local_addr, uint32_t len,
                                   const RemoteBuf& rb, uint32_t imm,
                                   uint64_t wr_id) {
  return post_rdma(IBV_WR_RDMA_WRITE_WITH_IMM, lkey, local_addr, len, rb, wr_id,
                   imm);
}

std::string RcConn::post_read(uint32_t lkey, void* local_addr, uint32_t len,
                              const RemoteBuf& rb, uint64_t wr_id) {
  return post_rdma(IBV_WR_RDMA_READ, lkey, local_addr, len, rb, wr_id);
}

std::string RcConn::post_atomic_fadd(uint32_t lkey, void* fetch_addr,
                                     const RemoteBuf& rb, uint64_t add,
                                     uint64_t wr_id) {
  struct ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uint64_t>(fetch_addr);
  sge.length = sizeof(uint64_t);
  sge.lkey = lkey;

  struct ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.atomic.remote_addr = rb.addr;
  wr.wr.atomic.rkey = rb.rkey;
  wr.wr.atomic.compare_add = add;
  return post_send_one(wr);
}

std::string RcConn::post_write_inline(const void* local_addr, uint32_t len,
                                      const RemoteBuf& rb, uint64_t wr_id) {
  struct ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  // Inline data: addr points at the payload, no lkey (the provider copies the
  // bytes into the WQE). const_cast is safe -- the send path only reads it.
  sge.addr = reinterpret_cast<uint64_t>(const_cast<void*>(local_addr));
  sge.length = len;

  struct ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = rb.addr;
  wr.wr.rdma.rkey = rb.rkey;
  return post_send_one(wr);
}

std::string RcConn::post_chain(struct ibv_send_wr* first) {
  struct ibv_send_wr* bad = nullptr;
  if (ibv_post_send(qp_, first, &bad) != 0)
    return errno_str("ibv_post_send(chain) failed");
  return "";
}

std::string RcConn::post_recv(uint32_t lkey, void* local_addr, uint32_t len,
                              uint64_t wr_id) {
  struct ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uint64_t>(local_addr);
  sge.length = len;
  sge.lkey = lkey;

  struct ibv_recv_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(qp_, &wr, &bad) != 0)
    return errno_str("ibv_post_recv failed");
  return "";
}

std::string RcConn::poll(int timeout_ms, struct ibv_wc* out_wc) {
  struct ibv_wc wc;
  // Poll once: sets *got, or *err on a hard error. Shared by both phases below.
  auto reap = [&](bool* got, std::string* err) -> bool {
    std::memset(&wc, 0, sizeof(wc));
    const int n = ibv_poll_cq(cq_, 1, &wc);
    if (n < 0) { *err = "ibv_poll_cq failed"; return false; }
    if (n == 0) { *got = false; return true; }
    *got = true;
    if (out_wc != nullptr) *out_wc = wc;
    if (wc.status != IBV_WC_SUCCESS) {
      const char* s = ibv_wc_status_str(wc.status);
      *err = std::string("completion error: ") + (s ? s : "unknown");
      return false;
    }
    return true;
  };

  // Phase 1: busy-spin for up to kSpinUs to catch the common sub-ms completion
  // without incurring the 1ms sleep granularity below (which floors every perf
  // measurement at ~1ms). Skipped when timeout_ms is small enough that the spin
  // itself could exceed the budget. The elapsed spin time seeds waited_us so the
  // total poll() duration stays within timeout_ms.
  constexpr long kSpinUs = 500;  // time-bound spin window, not iteration count
  long waited_us = 0;
  if (timeout_ms * 1000L > kSpinUs) {
    struct timespec spin_start, spin_now;
    ::clock_gettime(CLOCK_MONOTONIC, &spin_start);
    for (;;) {
      bool got = false;
      std::string err;
      if (!reap(&got, &err)) return err;
      if (got) return "";
      ::clock_gettime(CLOCK_MONOTONIC, &spin_now);
      const long elapsed =
          (spin_now.tv_sec - spin_start.tv_sec) * 1000000L +
          (spin_now.tv_nsec - spin_start.tv_nsec) / 1000L;
      if (elapsed >= kSpinUs) { waited_us = elapsed; break; }
    }
  }

  // Phase 2: sleep-poll the long tail within the remaining budget.
  const long timeout_us = static_cast<long>(timeout_ms) * 1000L;
  for (;;) {
    bool got = false;
    std::string err;
    if (!reap(&got, &err)) return err;
    if (got) return "";
    if (waited_us >= timeout_us)
      return "CQ poll timeout after " + std::to_string(timeout_ms) + "ms";
    sleep_us(1000);
    waited_us += 1000;
  }
}

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
