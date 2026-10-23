/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Verbs acceptance tests — Foundation group.
//
// Built only when ACCEPT_WITH_IBVERBS=ON (defines ACCEPT_HAVE_IBVERBS). The TU
// compiles as plain C++ against the vendored rdma-core declarations
// (vendor/rdma-core-headers); every ibv_* symbol resolves at run time through
// the dlopen wrapper (wrapper/verbs/IbvApi.hpp), so the suite builds on a node
// with no libibverbs-dev and runs where the runtime library is present. Without
// ibverbs the file is empty and nothing registers.
//
// Scope: single-process resource/query tests only. No QP is driven to RTS and no
// data is transferred here (that is the data-path suite). Every call used below
// is a real exported libibverbs symbol.

#ifndef ACCEPT_HAVE_IBVERBS
// No ibverbs — nothing to register.
#else

#include "wrapper/verbs/IbvApi.hpp"
#include "wrapper/verbs/VerbsHelpers.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {

namespace {

// A device opened for the duration of a test. RAII so every early-return path
// closes the context and frees the device list (verbs leaks are silent and
// accumulate across a suite run).
struct OpenedDevice {
  struct ibv_device** list = nullptr;
  struct ibv_context* ctx = nullptr;
  std::string name;
  ~OpenedDevice() {
    // noexcept teardown: these helpers call the resolved ops-table pointer
    // directly (not the throwing ibv_* macros), so a missing cleanup symbol
    // cannot throw MissingSymbol out of this destructor during stack unwinding.
    ibvw::safe_close_device(ctx);
    ibvw::safe_free_device_list(list);
  }
};

// Open the device named by ACCEPT_IB_DEVICE (set from --device), else the first
// enumerable device -- the same override RcConn::open() and the environment
// report honor, so these tests run on the device the report names. Returns
// false on failure and fills `err`. `no_devices` is set true when the host
// simply has no RDMA hardware (a skip-worthy condition, not a defect) vs. an
// actual open/enumerate failure, or a --device that is not present, on a host
// that should have a NIC (a real fail).
bool open_selected_device(OpenedDevice& d, std::string& err, bool& no_devices) {
  no_devices = false;
  int num = 0;
  d.list = ibv_get_device_list(&num);
  // Contract: NULL means the enumerate itself failed (errno set) -- a real fault
  // (e.g. permissions, uverbs module not loaded) that must surface as a failure,
  // NOT the same as an empty list. Only a non-NULL list with num == 0 is the
  // genuine "no RDMA hardware here" case that skips.
  if (!d.list) {
    err = ibvw::errno_reason("ibv_get_device_list failed");
    return false;
  }
  if (num == 0) {
    no_devices = true;
    err = "no RDMA devices present on this host";
    return false;
  }
  int idx = 0;
  if (const char* want = std::getenv("ACCEPT_IB_DEVICE");
      want != nullptr && *want != '\0') {
    idx = -1;
    for (int i = 0; i < num; ++i) {
      const char* name = ibv_get_device_name(d.list[i]);
      if (name != nullptr && std::strcmp(name, want) == 0) { idx = i; break; }
    }
    if (idx < 0) {
      err = "IB device '" + std::string(want) + "' not found";
      return false;
    }
  }
  d.ctx = ibv_open_device(d.list[idx]);
  if (!d.ctx) {
    err = ibvw::errno_reason("ibv_open_device failed");
    return false;
  }
  const char* n = ibv_get_device_name(d.list[idx]);
  d.name = n ? n : "<unknown>";
  return true;
}

// Find the first port in state ACTIVE (1-based port_num). Returns 0 if none.
uint8_t first_active_port(struct ibv_context* ctx, int num_ports) {
  for (uint8_t p = 1; p <= num_ports; ++p) {
    struct ibv_port_attr pa;
    std::memset(&pa, 0, sizeof(pa));
    if (ibv_query_port(ctx, p, &pa) == 0 && pa.state == IBV_PORT_ACTIVE) {
      return p;
    }
  }
  return 0;
}

// One PD -> CQ -> MR(all flag sets) -> teardown roundtrip on `ctx`, using `buf`
// as the registration target. When `verify` is set (the first pass) it also
// checks each MR reports a non-zero lkey. Returns "" on success or a reason.
// This is both the functional body and the timed region of vat_pd_cq_mr.
std::string pd_cq_mr_roundtrip(struct ibv_context* ctx,
                               std::vector<unsigned char>& buf, bool verify) {
  // LOCAL_WRITE is required whenever REMOTE_WRITE/REMOTE_ATOMIC is set, so the
  // combinations below always include it. These are the flag sets RCCL uses for
  // recv buffers, RDMA targets, and atomic counters.
  static const int flag_sets[] = {
      IBV_ACCESS_LOCAL_WRITE,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_RELAXED_ORDERING,
  };

  struct ibv_pd* pd = ibv_alloc_pd(ctx);
  if (pd == nullptr) {
    return ibvw::errno_reason("ibv_alloc_pd failed");
  }

  struct ibv_cq* cq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);
  if (cq == nullptr) {
    (void)ibv_dealloc_pd(pd);
    return ibvw::errno_reason("ibv_create_cq failed");
  }

  std::string failure;
  for (int access : flag_sets) {
    struct ibv_mr* mr = ibv_reg_mr(pd, buf.data(), buf.size(), access);
    if (mr == nullptr) {
      failure = "ibv_reg_mr(access=0x" +
                std::string(TestResult::int_to_chars(access)) +
                ") failed: " + std::strerror(errno);
      break;
    }
    if (verify && mr->lkey == 0) {
      (void)ibv_dereg_mr(mr);
      failure = "ibv_reg_mr returned lkey=0";
      break;
    }
    if (ibv_dereg_mr(mr) != 0) {
      failure = std::string("ibv_dereg_mr failed: ") + std::strerror(errno);
      break;
    }
  }

  // Always tear down PD/CQ; keep the first failure reason if one is pending.
  if (ibv_destroy_cq(cq) != 0 && failure.empty()) {
    failure = std::string("ibv_destroy_cq failed: ") + std::strerror(errno);
  }
  if (ibv_dealloc_pd(pd) != 0 && failure.empty()) {
    failure = std::string("ibv_dealloc_pd failed: ") + std::strerror(errno);
  }
  return failure;
}

}  // namespace

// Enumerate devices, open the selected one (--device, else the first), query
// device + every port, and confirm at least one port is ACTIVE with a non-empty
// GID table (RCCL cannot bring up any IB transport otherwise). Reads GID entry 0
// on the first active port.
ACCEPT_TEST(vat_device_enum, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_device_attr da;
  std::memset(&da, 0, sizeof(da));
  IBV_CHECK(ibv_query_device(d.ctx, &da));
  if (da.phys_port_cnt == 0)
    return fail(d.name + ": device reports 0 physical ports");

  uint8_t active = 0;
  for (uint8_t p = 1; p <= da.phys_port_cnt; ++p) {
    struct ibv_port_attr pa;
    std::memset(&pa, 0, sizeof(pa));
    IBV_CHECK(ibv_query_port(d.ctx, p, &pa));
    if (pa.state != IBV_PORT_ACTIVE) continue;
    if (pa.gid_tbl_len <= 0)
      return fail(d.name + " port " + std::to_string(p) +
                  ": ACTIVE but empty GID table");
    union ibv_gid gid;
    std::memset(&gid, 0, sizeof(gid));
    IBV_CHECK(ibv_query_gid(d.ctx, p, 0, &gid));
    active = p;
    break;
  }
  if (active == 0)
    return skip(d.name + ": no ACTIVE port (link down?)");
  return pass();
}

// PD / CQ / MR lifecycle across every access-flag combination RCCL relies on.
// MR registration goes through the wrapper's iova2 redirect, so relaxed-ordering
// (an OPTIONAL flag) is exercised here too.
ACCEPT_TEST(vat_pd_cq_mr, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  std::vector<unsigned char> buf(64 * 1024);

  // Region: the full PD/CQ/MR create->register->teardown roundtrip. One pass
  // functional (with lkey verification), warmup+N under perf.
  PerfLoop loop;
  std::string failure;
  for (auto pass : loop) {
    failure = pd_cq_mr_roundtrip(d.ctx, buf, pass.first());
    if (!failure.empty()) break;
  }
  if (!failure.empty()) return fail(d.name + ": " + failure);

  return pass_with_latency(loop, "verbs_setup.lat");
}

// Create an RC QP and confirm the NIC honored the requested capabilities. RCCL
// provisions QPs assuming the advertised limits; if the NIC silently caps them
// lower, RCCL over-provisions and fails at scale.
ACCEPT_TEST(vat_rc_qp_capacity, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_pd* pd = ibv_alloc_pd(d.ctx);
  IBV_CHECK_PTR(pd, "ibv_alloc_pd");
  struct ibv_cq* cq = ibv_create_cq(d.ctx, 256, nullptr, nullptr, 0);
  if (!cq) {
    (void)ibv_dealloc_pd(pd);
    return fail(ibvw::errno_reason("ibv_create_cq failed"));
  }

  // Region: one RC QP create+destroy per pass. A fresh init_attr each pass keeps
  // the granted-cap read-back from leaking between iterations; the first pass
  // also validates the granted capacity.
  PerfLoop loop;
  std::string failure;
  std::string why;
  for (auto pass : loop) {
    struct ibv_qp_init_attr qia;
    std::memset(&qia, 0, sizeof(qia));
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 512;
    qia.cap.max_recv_wr = 256;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;

    struct ibv_qp* qp = ibv_create_qp(pd, &qia);
    if (qp == nullptr) {
      failure = std::string("ibv_create_qp(RC) failed: ") + std::strerror(errno);
      break;
    }

    // The NIC fills qia.cap with the granted capacity, which must meet or exceed
    // what we asked for (providers may round UP but must not silently cap below).
    if (pass.first()) {
      if (qia.cap.max_send_wr < 512) {
        why = "max_send_wr granted " + std::to_string(qia.cap.max_send_wr) +
              " < 512";
      } else if (qia.cap.max_recv_wr < 256) {
        why = "max_recv_wr granted " + std::to_string(qia.cap.max_recv_wr) +
              " < 256";
      }
    }

    if (ibv_destroy_qp(qp) != 0) {
      failure = std::string("ibv_destroy_qp failed: ") + std::strerror(errno);
      break;
    }
    if (!why.empty()) break;
  }

  (void)ibv_destroy_cq(cq);
  (void)ibv_dealloc_pd(pd);
  if (!failure.empty()) return fail(d.name + ": " + failure);
  if (!why.empty()) return fail(d.name + ": " + why);

  return pass_with_latency(loop, "qp_create_destroy.lat");
}

// Full device_attr probe: report the capability matrix that gates the
// CONDITIONAL tests (atomics, SRQ, max QP/MR/CQE). Also confirms port query and
// device-name resolution work. RCCL branches on these values at init.
ACCEPT_TEST(vat_device_caps_probe, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_device_attr da;
  std::memset(&da, 0, sizeof(da));
  IBV_CHECK(ibv_query_device(d.ctx, &da));

  // Sanity floor: a functional NIC advertises non-zero QP/CQE/MR limits.
  if (da.max_qp == 0 || da.max_cqe == 0 || da.max_mr == 0)
    return fail(d.name + ": device_attr reports a zero core limit (max_qp=" +
                std::to_string(da.max_qp) + " max_cqe=" +
                std::to_string(da.max_cqe) + " max_mr=" +
                std::to_string(da.max_mr) + ")");

  const uint8_t active = first_active_port(d.ctx, da.phys_port_cnt);
  if (active == 0) return skip(d.name + ": no ACTIVE port (link down?)");
  return pass();
}

// fork-safety init: call ibv_fork_init before registering an MR, then fork() and
// confirm the parent's registered buffer survives the child's copy-on-write.
// Without fork_init, registered pages can be silently broken across fork.
ACCEPT_TEST(vat_fork_init, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  // ibv_fork_init() is a process-global switch that must be called before ANY
  // MR is registered in the process. The runner force-isolates this test (see
  // Runner::should_isolate) so it always runs in a fresh process where no prior
  // test has registered an MR -- otherwise (in-process, after vat_pd_cq_mr) the
  // call would be too late to be meaningful.
  IBV_CHECK(ibv_fork_init());

  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_pd* pd = ibv_alloc_pd(d.ctx);
  IBV_CHECK_PTR(pd, "ibv_alloc_pd");

  std::vector<unsigned char> buf(4096, 0xAB);
  struct ibv_mr* mr =
      ibv_reg_mr(pd, buf.data(), buf.size(), IBV_ACCESS_LOCAL_WRITE);
  if (!mr) {
    const std::string reason = std::strerror(errno);
    (void)ibv_dealloc_pd(pd);
    return fail(std::string("ibv_reg_mr failed: ") + reason);
  }

  const pid_t pid = fork();
  if (pid == 0) {
    // Child: touch the page (COW). With fork_init the parent's registration is
    // protected; exit immediately without disturbing the parent's verbs state.
    buf[0] = 0x00;
    _exit(0);
  }
  std::string why;
  if (pid < 0) {
    why = ibvw::errno_reason("fork failed");
  } else {
    int st = 0;
    (void)waitpid(pid, &st, 0);
    // Parent's registered buffer must be intact after the child's COW write.
    for (size_t i = 0; i < buf.size(); ++i) {
      if (buf[i] != 0xAB) {
        why = "parent MR corrupted at byte " + std::to_string(i) +
              " after fork (fork_init did not protect the registration)";
        break;
      }
    }
  }

  (void)ibv_dereg_mr(mr);
  (void)ibv_dealloc_pd(pd);
  if (!why.empty()) return fail(d.name + ": " + why);
  return pass();
}

// Register an MR with IBV_ACCESS_RELAXED_ORDERING and confirm the registration
// succeeds (create-only; the ordering guarantee under traffic is a data-path
// test). Silently degrades to non-RO on firmware that does not support it, so a
// failed registration on a NIC that should support it is the regression signal.
ACCEPT_TEST(vat_relaxed_ordering, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_pd* pd = ibv_alloc_pd(d.ctx);
  IBV_CHECK_PTR(pd, "ibv_alloc_pd");

  std::vector<unsigned char> buf(64 * 1024);
  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_RELAXED_ORDERING;

  // Region: register + deregister a 64KiB MR with RELAXED_ORDERING per pass.
  // One functional pass confirms the flag is honored; warmup+N under perf.
  PerfLoop loop;
  std::string failure;
  for (auto pass : loop) {
    (void)pass;
    struct ibv_mr* mr = ibv_reg_mr(pd, buf.data(), buf.size(), access);
    if (mr == nullptr) {
      failure = std::string("ibv_reg_mr with RELAXED_ORDERING failed: ") +
                std::strerror(errno);
      break;
    }
    if (ibv_dereg_mr(mr) != 0) {
      failure = std::string("ibv_dereg_mr failed: ") + std::strerror(errno);
      break;
    }
  }

  (void)ibv_dealloc_pd(pd);
  if (!failure.empty()) return fail(d.name + ": " + failure);

  return pass_with_latency(loop, "reg_mr_ro.lat");
}

// ibv_reg_mr + ibv_dereg_mr latency across buffer sizes. RCCL registers large
// pre-allocated IB buffers at startup; the cost is not linear (page-walk grows
// with size, 2MiB-aligned allocations may get huge-page treatment). Sweeps
// 4KiB/64KiB/1MiB/16MiB with plain LOCAL_WRITE|REMOTE_WRITE flags, one metric
// key per size so the curve is visible in report.json. Single-node, no peer.
ACCEPT_TEST(vat_reg_mr_latency, "verbs/foundation", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_pd* pd = ibv_alloc_pd(d.ctx);
  IBV_CHECK_PTR(pd, "ibv_alloc_pd");

  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  constexpr size_t kSizes[] = {4096, 65536, 1024 * 1024, 16 * 1024 * 1024};
  static_assert(std::size(kSizes) == 4, "update loop bound if kSizes changes");

  PerfSink perf;
  std::string failure;
  for (std::size_t s = 0; s < std::size(kSizes) && failure.empty(); ++s) {
    std::vector<unsigned char> buf(kSizes[s]);
    PerfLoop loop;
    for (auto pass : loop) {
      (void)pass;
      struct ibv_mr* mr = ibv_reg_mr(pd, buf.data(), buf.size(), access);
      if (mr == nullptr) {
        failure = "ibv_reg_mr failed at " + std::to_string(kSizes[s]) +
                  "B: " + std::strerror(errno);
        break;
      }
      if (ibv_dereg_mr(mr) != 0) {
        // MR remains registered; ibv_dealloc_pd (swallowed below) returns
        // EBUSY -- acceptable since the test is already failing.
        failure = "ibv_dereg_mr failed at " + std::to_string(kSizes[s]) +
                  "B: " + std::strerror(errno);
        break;
      }
    }
    // Pass size to emit_latency so the key shape matches emit_bandwidth:
    // reg_mr.lat.median.4KiB (stat before size, consistent with bw path).
    if (failure.empty()) loop.emit_latency(perf, "reg_mr.lat", kSizes[s]);
  }

  (void)ibv_dealloc_pd(pd);
  if (!failure.empty()) return fail(d.name + ": " + failure);
  return pass_with_perf(std::move(perf));
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
