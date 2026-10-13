/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#ifdef ACCEPT_HAVE_IBVERBS

#include "verbs/UdConn.hpp"

#include <arpa/inet.h>  // htonl
#include <time.h>       // nanosleep

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "common/Capability.hpp"

namespace acceptance {
namespace verbsx {

namespace {

constexpr int kCqDepth = 32;
// A fixed QKEY shared by both endpoints. UD delivery requires matching QKEYs.
// Matches RCCL's NCCL_IB_RECOVERY_QKEY (net_ib_cast/p2p_resiliency_recovery_
// internal.h) so this models the real recovery-datagram QP exactly.
constexpr uint32_t kUdQkey = 0x1234ABCD;

int roce_v2_gid_index(const std::string& dev) {
  for (const auto& d : environment().rdma_devices)
    if (d.device == dev && d.gid_index >= 0) return d.gid_index;
  return -1;
}

void sleep_us(long us) {
  struct timespec ts;
  ts.tv_sec = us / 1000000L;
  ts.tv_nsec = (us % 1000000L) * 1000L;
  ::nanosleep(&ts, nullptr);
}

std::string errno_str(const char* what) { return ibvw::errno_reason(what); }

}  // namespace

UdConn::~UdConn() {
  ibvw::safe_destroy_qp(qp_);
  ibvw::safe_destroy_cq(cq_);
  ibvw::safe_dealloc_pd(pd_);
  ibvw::safe_close_device(ctx_);
}

std::string UdConn::open(const std::string& dev_name, bool& no_device) {
  no_device = false;
  // Empty dev_name means "auto-select"; honor ACCEPT_IB_DEVICE (set from
  // --device) first, the same override Capability::detect() applies to the
  // reported environment/GID lookup, so --device actually steers the connect
  // path instead of only the reported vendor/GID.
  std::string target = dev_name;
  if (target.empty()) {
    if (const char* want = std::getenv("ACCEPT_IB_DEVICE");
        want != nullptr && *want != '\0') {
      target = want;
    }
  }
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
    // Mirror the auto-select branch's port check: a named device with a down
    // port should surface a clear reason here rather than failing opaquely at
    // QP bring-up. (RcConn::open historically skipped this on the named path.)
    if (ctx_ != nullptr) {
      struct ibv_port_attr pa;
      std::memset(&pa, 0, sizeof(pa));
      if (ibv_query_port(ctx_, ib_port_, &pa) != 0 ||
          pa.state != IBV_PORT_ACTIVE) {
        ibvw::safe_close_device(ctx_);
        ctx_ = nullptr;
        ibvw::safe_free_device_list(list);
        return "IB device '" + target + "' has no ACTIVE port";
      }
    }
  } else {
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
      no_device = true;
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

  gid_index_ = roce_v2_gid_index(dev_name_);

  struct ibv_port_attr pa;
  std::memset(&pa, 0, sizeof(pa));
  if (ibv_query_port(ctx_, ib_port_, &pa) != 0)
    return errno_str("ibv_query_port failed");
  local_.lid = pa.lid;

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

  return create_qp_init();
}

std::string UdConn::create_qp_init() {
  struct ibv_qp_init_attr qia;
  std::memset(&qia, 0, sizeof(qia));
  qia.send_cq = cq_;
  qia.recv_cq = cq_;
  qia.qp_type = IBV_QPT_UD;
  qia.sq_sig_all = 0;
  qia.cap.max_send_wr = 16;
  qia.cap.max_recv_wr = 16;
  qia.cap.max_send_sge = 1;
  qia.cap.max_recv_sge = 1;
  qia.cap.max_inline_data = 0;
  qp_ = ibv_create_qp(pd_, &qia);
  if (qp_ == nullptr) return errno_str("ibv_create_qp(UD) failed");

  // RESET -> INIT: UD sets pkey_index, port, and QKEY (no access flags).
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_INIT;
  qa.pkey_index = 0;
  qa.port_num = static_cast<uint8_t>(ib_port_);
  qa.qkey = kUdQkey;
  if (ibv_modify_qp(qp_, &qa,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_QKEY) != 0)
    return errno_str("ibv_modify_qp INIT(UD) failed");

  local_.qpn = qp_->qp_num;
  local_.qkey = kUdQkey;
  return "";
}

std::string UdConn::to_rts() {
  // INIT -> RTR: UD needs only the state.
  struct ibv_qp_attr qa;
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp_, &qa, IBV_QP_STATE) != 0)
    return errno_str("ibv_modify_qp RTR(UD) failed");

  // RTR -> RTS: UD needs state + sq_psn.
  std::memset(&qa, 0, sizeof(qa));
  qa.qp_state = IBV_QPS_RTS;
  qa.sq_psn = 0;
  if (ibv_modify_qp(qp_, &qa, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0)
    return errno_str("ibv_modify_qp RTS(UD) failed");
  return "";
}

std::string UdConn::recreate() {
  ibvw::safe_destroy_qp(qp_);
  qp_ = nullptr;
  return create_qp_init();
}

struct ibv_ah* UdConn::create_ah(const UdInfo& peer) {
  struct ibv_ah_attr aa;
  std::memset(&aa, 0, sizeof(aa));
  aa.is_global = 1;
  aa.port_num = static_cast<uint8_t>(ib_port_);
  aa.dlid = peer.lid;
  aa.sl = 0;
  aa.src_path_bits = 0;
  std::memcpy(aa.grh.dgid.raw, peer.gid, 16);
  aa.grh.sgid_index = static_cast<uint8_t>(gid_index_);
  aa.grh.hop_limit = 64;
  return ibv_create_ah(pd_, &aa);
}

std::string UdConn::post_send_imm(struct ibv_ah* ah, const UdInfo& peer,
                                  uint32_t lkey, void* addr, uint32_t len,
                                  uint32_t imm, uint64_t wr_id) {
  struct ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uint64_t>(addr);
  sge.length = len;
  sge.lkey = lkey;

  struct ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(imm);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.ud.ah = ah;
  wr.wr.ud.remote_qpn = peer.qpn;
  wr.wr.ud.remote_qkey = peer.qkey;

  struct ibv_send_wr* bad = nullptr;
  if (ibv_post_send(qp_, &wr, &bad) != 0)
    return errno_str("ibv_post_send(UD) failed");
  return "";
}

std::string UdConn::post_recv(uint32_t lkey, void* addr, uint32_t len,
                              uint64_t wr_id) {
  struct ibv_sge sge;
  std::memset(&sge, 0, sizeof(sge));
  // The provider writes the 40-byte GRH ahead of the payload, so the posted
  // buffer spans len + kUdGrhLen.
  sge.addr = reinterpret_cast<uint64_t>(addr);
  sge.length = len + kUdGrhLen;
  sge.lkey = lkey;

  struct ibv_recv_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(qp_, &wr, &bad) != 0)
    return errno_str("ibv_post_recv(UD) failed");
  return "";
}

std::string UdConn::poll(int timeout_ms, struct ibv_wc* out_wc) {
  struct ibv_wc wc;
  long waited_us = 0;
  const long timeout_us = static_cast<long>(timeout_ms) * 1000L;
  for (;;) {
    std::memset(&wc, 0, sizeof(wc));
    const int n = ibv_poll_cq(cq_, 1, &wc);
    if (n < 0) return "ibv_poll_cq failed";
    if (n > 0) {
      if (out_wc != nullptr) *out_wc = wc;
      if (wc.status != IBV_WC_SUCCESS) {
        const char* s = ibv_wc_status_str(wc.status);
        return std::string("completion error: ") + (s ? s : "unknown");
      }
      return "";
    }
    if (waited_us >= timeout_us)
      return "CQ poll timeout after " + std::to_string(timeout_ms) + "ms";
    sleep_us(1000);
    waited_us += 1000;
  }
}

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
