/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Verbs (libibverbs) dlopen ops-table wrapper. Same model as the HIP/HSA
// wrappers: compile against the vendored rdma-core headers
// (vendor/rdma-core-headers), link nothing, resolve every ibv_* entry point at
// run time through wrap::load_library. A symbol absent on the running
// libibverbs makes the dependent test skip (checked_op -> MissingSymbol) rather
// than failing the load.
//
// Scope: the control-path ibv_* entry points. Every entry point below is a REAL
// exported libibverbs symbol (verified with nm -D). The data-path calls
// (ibv_post_send/recv, ibv_poll_cq) are `static inline` dispatching through
// qp->context->ops and are NOT dlsym-able, so they are not in this table;
// RcConn calls them directly once ibv_open_device has filled the ops table.
//
// Two header quirks handled here (see vendor README):
//   * ibv_reg_mr is a compile-time macro picking ibv_reg_mr / ibv_reg_mr_iova2.
//     We bind the exported ibv_reg_mr_iova2 and redirect ibv_reg_mr(...) to it
//     (iova = addr), which also covers IBV_ACCESS_RELAXED_ORDERING (an OPTIONAL
//     flag that requires the iova2 path).
//   * ibv_query_port is a macro over a header inline (___ibv_query_port) that we
//     must not instantiate; the plain exported symbol is old-ABI. We bind it via
//     an explicit function-pointer typedef and redirect the macro to it.

#include <infiniband/verbs.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "wrapper/DynLib.hpp"

namespace acceptance {
namespace ibvw {

// Explicit signature for the exported (old-ABI) ibv_query_port. The symbol fills
// the classic prefix of ibv_port_attr; callers zero the struct first so the
// (unused-by-old-ABI) trailing fields stay defined. Typedef'd because the
// header spells ibv_query_port as a macro.
using ibv_query_port_fn = int (*)(struct ibv_context*, uint8_t,
                                  struct ibv_port_attr*);

typedef struct ibv_lib_ops_t {
  decltype(::ibv_get_device_list)* ibv_get_device_list;
  decltype(::ibv_free_device_list)* ibv_free_device_list;
  decltype(::ibv_open_device)* ibv_open_device;
  decltype(::ibv_close_device)* ibv_close_device;
  decltype(::ibv_get_device_name)* ibv_get_device_name;
  decltype(::ibv_query_device)* ibv_query_device;
  ibv_query_port_fn ibv_query_port;
  decltype(::ibv_query_gid)* ibv_query_gid;
  decltype(::ibv_alloc_pd)* ibv_alloc_pd;
  decltype(::ibv_dealloc_pd)* ibv_dealloc_pd;
  decltype(::ibv_create_cq)* ibv_create_cq;
  decltype(::ibv_destroy_cq)* ibv_destroy_cq;
  decltype(::ibv_reg_mr_iova2)* ibv_reg_mr_iova2;
  decltype(::ibv_reg_dmabuf_mr)* ibv_reg_dmabuf_mr;
  decltype(::ibv_dereg_mr)* ibv_dereg_mr;
  decltype(::ibv_create_qp)* ibv_create_qp;
  decltype(::ibv_destroy_qp)* ibv_destroy_qp;
  decltype(::ibv_modify_qp)* ibv_modify_qp;
  decltype(::ibv_query_qp)* ibv_query_qp;
  decltype(::ibv_fork_init)* ibv_fork_init;
  decltype(::ibv_wc_status_str)* ibv_wc_status_str;
  decltype(::ibv_create_ah)* ibv_create_ah;
  decltype(::ibv_destroy_ah)* ibv_destroy_ah;
  decltype(::ibv_get_async_event)* ibv_get_async_event;
  decltype(::ibv_ack_async_event)* ibv_ack_async_event;
  decltype(::ibv_event_type_str)* ibv_event_type_str;
  decltype(::ibv_create_srq)* ibv_create_srq;
  decltype(::ibv_destroy_srq)* ibv_destroy_srq;
  decltype(::ibv_set_ece)* ibv_set_ece;
  decltype(::ibv_query_ece)* ibv_query_ece;
} ibv_lib_ops_t;

// Order MUST match the struct member order above (load_library fills by index).
static std::vector<std::string> ibv_fn_names = {
    "ibv_get_device_list",
    "ibv_free_device_list",
    "ibv_open_device",
    "ibv_close_device",
    "ibv_get_device_name",
    "ibv_query_device",
    "ibv_query_port",
    "ibv_query_gid",
    "ibv_alloc_pd",
    "ibv_dealloc_pd",
    "ibv_create_cq",
    "ibv_destroy_cq",
    "ibv_reg_mr_iova2",
    "ibv_reg_dmabuf_mr",
    "ibv_dereg_mr",
    "ibv_create_qp",
    "ibv_destroy_qp",
    "ibv_modify_qp",
    "ibv_query_qp",
    "ibv_fork_init",
    "ibv_wc_status_str",
    "ibv_create_ah",
    "ibv_destroy_ah",
    "ibv_get_async_event",
    "ibv_ack_async_event",
    "ibv_event_type_str",
    "ibv_create_srq",
    "ibv_destroy_srq",
    "ibv_set_ece",
    "ibv_query_ece",
};

extern ibv_lib_ops_t ibv_lib_ops;

// Per-symbol call-site guard: yields the pointer when present, throws
// MissingSymbol{"name"} (-> Skip at the test boundary) when the running
// libibverbs lacks it.
#define IBVW_OP(name) \
  (::acceptance::wrap::checked_op(::acceptance::ibvw::ibv_lib_ops.name, #name))

// reg_mr redirect: route the 4-arg ibv_reg_mr(...) through the exported iova2
// entry point (iova = addr). Goes through checked_op so a missing symbol skips.
inline struct ibv_mr* reg_mr_via_iova2(struct ibv_pd* pd, void* addr,
                                       size_t length, unsigned int access) {
  return ::acceptance::wrap::checked_op(ibv_lib_ops.ibv_reg_mr_iova2,
                                        "ibv_reg_mr_iova2")(
      pd, addr, length, reinterpret_cast<uint64_t>(addr), access);
}

// noexcept teardown helpers for RAII destructors. Unlike the ibv_* macros these
// call the resolved ops-table pointer DIRECTLY and null-guard it, so they never
// throw MissingSymbol -- a destructor throwing during stack unwinding (e.g. the
// test body already unwinding from a MissingSymbol) would std::terminate. Safe
// to call with a null handle. Defined here (before the #define block) so the raw
// member names are still usable.
inline void safe_close_device(struct ibv_context* ctx) noexcept {
  if (ctx && ibv_lib_ops.ibv_close_device) ibv_lib_ops.ibv_close_device(ctx);
}
inline void safe_free_device_list(struct ibv_device** list) noexcept {
  if (list && ibv_lib_ops.ibv_free_device_list)
    ibv_lib_ops.ibv_free_device_list(list);
}
inline void safe_destroy_qp(struct ibv_qp* qp) noexcept {
  if (qp && ibv_lib_ops.ibv_destroy_qp) ibv_lib_ops.ibv_destroy_qp(qp);
}
inline void safe_destroy_cq(struct ibv_cq* cq) noexcept {
  if (cq && ibv_lib_ops.ibv_destroy_cq) ibv_lib_ops.ibv_destroy_cq(cq);
}
inline void safe_dealloc_pd(struct ibv_pd* pd) noexcept {
  if (pd && ibv_lib_ops.ibv_dealloc_pd) ibv_lib_ops.ibv_dealloc_pd(pd);
}
inline void safe_dereg_mr(struct ibv_mr* mr) noexcept {
  if (mr && ibv_lib_ops.ibv_dereg_mr) ibv_lib_ops.ibv_dereg_mr(mr);
}
inline void safe_destroy_srq(struct ibv_srq* srq) noexcept {
  if (srq && ibv_lib_ops.ibv_destroy_srq) ibv_lib_ops.ibv_destroy_srq(srq);
}
inline void safe_destroy_ah(struct ibv_ah* ah) noexcept {
  if (ah && ibv_lib_ops.ibv_destroy_ah) ibv_lib_ops.ibv_destroy_ah(ah);
}

// Format "<what>: <errno string>" for a failed libibverbs control-path call
// (libibverbs reports failures through errno). Shared by RcConn / GdrMem / the
// verbs Foundation tests so the message shape stays uniform.
inline std::string errno_reason(const char* what) {
  return std::string(what) + ": " + std::strerror(errno);
}

#define ibv_get_device_list IBVW_OP(ibv_get_device_list)
#define ibv_free_device_list IBVW_OP(ibv_free_device_list)
#define ibv_open_device IBVW_OP(ibv_open_device)
#define ibv_close_device IBVW_OP(ibv_close_device)
#define ibv_get_device_name IBVW_OP(ibv_get_device_name)
#define ibv_query_device IBVW_OP(ibv_query_device)
#undef ibv_query_port
#define ibv_query_port IBVW_OP(ibv_query_port)
#define ibv_query_gid IBVW_OP(ibv_query_gid)
#define ibv_alloc_pd IBVW_OP(ibv_alloc_pd)
#define ibv_dealloc_pd IBVW_OP(ibv_dealloc_pd)
#define ibv_create_cq IBVW_OP(ibv_create_cq)
#define ibv_destroy_cq IBVW_OP(ibv_destroy_cq)
#define ibv_reg_mr_iova2 IBVW_OP(ibv_reg_mr_iova2)
#undef ibv_reg_mr
#define ibv_reg_mr(pd, addr, length, access) \
  (::acceptance::ibvw::reg_mr_via_iova2((pd), (addr), (length), (access)))
#define ibv_dereg_mr IBVW_OP(ibv_dereg_mr)
#define ibv_create_qp IBVW_OP(ibv_create_qp)
#define ibv_destroy_qp IBVW_OP(ibv_destroy_qp)
#define ibv_modify_qp IBVW_OP(ibv_modify_qp)
#define ibv_query_qp IBVW_OP(ibv_query_qp)
#define ibv_fork_init IBVW_OP(ibv_fork_init)
// ibv_reg_dmabuf_mr / ibv_wc_status_str are plain exported symbols (no header
// macro), so the redirect is the straight per-symbol guard.
#define ibv_reg_dmabuf_mr IBVW_OP(ibv_reg_dmabuf_mr)
#define ibv_wc_status_str IBVW_OP(ibv_wc_status_str)
#define ibv_create_ah IBVW_OP(ibv_create_ah)
#define ibv_destroy_ah IBVW_OP(ibv_destroy_ah)
#define ibv_get_async_event IBVW_OP(ibv_get_async_event)
#define ibv_ack_async_event IBVW_OP(ibv_ack_async_event)
#define ibv_event_type_str IBVW_OP(ibv_event_type_str)
// SRQ + ECE are plain exported symbols (no header macro). ibv_post_srq_recv is a
// static-inline ops-table dispatch (like ibv_post_recv) and is called directly.
#define ibv_create_srq IBVW_OP(ibv_create_srq)
#define ibv_destroy_srq IBVW_OP(ibv_destroy_srq)
#define ibv_set_ece IBVW_OP(ibv_set_ece)
#define ibv_query_ece IBVW_OP(ibv_query_ece)

// True when `category` names a verbs test but libibverbs itself did not load
// (no RDMA stack on this host). The runner skips the whole category with one
// clean reason instead of faulting on a null ops table.
bool verbs_category_unavailable(const std::string& category);

// Loaded once at start-up (see main.cc). Returns true if libibverbs opened.
bool ibv_api_init();
void ibv_api_fini();
bool ibv_lib_loaded();

}  // namespace ibvw
}  // namespace acceptance
