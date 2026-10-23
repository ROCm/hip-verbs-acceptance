/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — IPC (legacy hipIpc* handle path) group.
//
// Mirrors RCCL's p2pSendSetup/p2pRecvSetup: export a 64-byte handle, ship it to
// a peer, import with lazy-enable-peer-access. The test fork()s a child before
// touching HIP (so each process inits the runtime independently; the category is
// force-isolated for this reason). Handle + sync flags travel through an
// anonymous MAP_SHARED region (flags via std::atomic_ref); the importer verifies
// with a real kernel read. All waits are bounded so a wedged peer fails, not
// hangs. Built only when ACCEPT_WITH_HIP=ON.
//
// The density tests add ROCM-29090 coverage: several processes per GPU
// exchanging handles across devices, the only shape that reproduces it.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <new>
#include <string>
#include <vector>

#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common/Perf.hpp"
#include "common/PosixUtil.hpp"
#include "common/TestRegistry.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// FillArgs (parent write) and MismArgs (child verify) are shared in HipTest.hpp.

// Cross-process state in a MAP_SHARED region;
//  plain int flags accessed via std::atomic_ref for ordering.
struct IpcShared {
  int handle_ready;          // exporter -> importer: handle is populated
  int child_done;            // importer -> exporter: import+verify finished
  int child_ok;              // importer result: 1 = pattern verified
  hipIpcMemHandle_t handle;  // byte-copyable 64-byte IPC handle
};

// Importer (child) exit codes, surfaced in the parent's failure message.
enum ImporterExit {
  kSetDeviceFailed = 10,
  kHandleWaitTimeout = 11,
  kOpenHandleFailed = 12,
  kVerifyFailed = 13,
};

// Bounded acquire-wait on a shared flag: true if it became non-zero
//  within `timeout_ms`, false on timeout (1 ms polls).
bool wait_flag(int* flag, int timeout_ms) {
  std::atomic_ref<int> a(*flag);
  for (int waited = 0; waited < timeout_ms; ++waited) {
    if (a.load(std::memory_order_acquire) != 0) return true;
    struct timespec req {};
    req.tv_sec = 0;
    req.tv_nsec = 1'000'000L;  // 1 ms
    (void)nanosleep(&req, nullptr);
  }
  return a.load(std::memory_order_acquire) != 0;
}

// Grace period before the wedged importer is SIGKILLed.
constexpr long kReapGraceMs = 2000;

void publish_flag(int* flag) {
  std::atomic_ref<int> a(*flag);
  a.store(1, std::memory_order_release);
}

// Kernel-verify the imported mapping on-device; true only
// if the pattern reads back intact (RAII frees the counter).
bool ipc_kernel_verify_pattern(const void* d_in, int n) {
  DeviceBuffer mismatches;  // single-int on-device mismatch counter
  if (mismatches.allocate(sizeof(int)) != hipSuccess) {
    return false;
  }
  if (hipMemset(mismatches.get(), 0, sizeof(int)) != hipSuccess) {
    return false;
  }

  MismArgs mism_args{static_cast<const int*>(d_in), n, mismatches.as<int>()};
  std::string launch_err;
  if (hip_kernel_launch("k_count_mismatch_indices", grid_for(n),
                        dim3(kBlockSize), 0, 0, &mism_args, sizeof(mism_args),
                        &launch_err) != hipSuccess) {
    return false;
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    return false;
  }

  int host_count = -1;
  if (hipMemcpy(&host_count, mismatches.get(), sizeof(int),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return false;
  }
  return host_count == 0;
}

// Child (importer): open the parent's IPC handle and kernel-verify the pattern
// through the mapping, then _exit with a precise code (never returns).
void run_ipc_importer_child(IpcShared* sh, int n, int wait_ms) {
  auto child_exit = [&](int code) {
    sh->child_ok = (code == 0) ? 1 : 0;
    publish_flag(&sh->child_done);
    _exit(code);
  };

  if (hipSetDevice(gpu()) != hipSuccess) {
    child_exit(kSetDeviceFailed);
  }
  if (!wait_flag(&sh->handle_ready, wait_ms)) {
    child_exit(kHandleWaitTimeout);
  }

  void* d_in = nullptr;
  if (hipIpcOpenMemHandle(&d_in, sh->handle, hipIpcMemLazyEnablePeerAccess) !=
      hipSuccess) {
    child_exit(kOpenHandleFailed);
  }

  const bool verified = ipc_kernel_verify_pattern(d_in, n);
  (void)hipIpcCloseMemHandle(d_in);
  child_exit(verified ? 0 : kVerifyFailed);
}

// Parent (exporter): stage the pattern, export the IPC handle (timed), hand it
// to the importer and bounded-wait for its verified result. Returns "" on pass.
std::string run_ipc_exporter_parent(IpcShared* sh, DeviceBuffer& buf,
                                    Stream& stream, int n, size_t bytes,
                                    int wait_ms, PerfSink& perf_sink) {
  // Phase 1: bring up device, buffer and stream.
  if (hipSetDevice(gpu()) != hipSuccess) {
    return "parent hipSetDevice failed";
  }
  if (buf.allocate(bytes) != hipSuccess) {
    return "parent hipMalloc failed";
  }
  if (stream.create() != hipSuccess) {
    return "parent hipStreamCreate failed";
  }

  // Phase 2: stage the index pattern into the buffer with a kernel write.
  FillArgs fill_args{buf.as<int>(), n};
  std::string launch_err;
  if (hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0,
                        stream.get(), &fill_args, sizeof(fill_args),
                        &launch_err) != hipSuccess) {
    return "parent kernel write failed: " + launch_err;
  }
  if (hipStreamSynchronize(stream.get()) != hipSuccess) {
    return "parent hipStreamSynchronize failed";
  }

  // Phase 3: export the IPC handle into shared memory (timed under perf mode).
  PerfLoop loop;
  hipError_t handle_err = hipSuccess;
  for (auto pass : loop) {
    handle_err = hipIpcGetMemHandle(&sh->handle, buf.get());
    if (handle_err != hipSuccess) {
      break;
    }
  }
  if (handle_err != hipSuccess) {
    return std::string("hipIpcGetMemHandle: ") + hipGetErrorString(handle_err);
  }
  loop.emit_latency(perf_sink, "ipc.get_handle");

  // Phase 4: signal the handle is ready (release), then bounded-wait for the
  // importer and read back its verified result.
  publish_flag(&sh->handle_ready);
  if (!wait_flag(&sh->child_done, wait_ms)) {
    return "importer did not finish within " + std::to_string(wait_ms) + " ms";
  }
  if (sh->child_ok != 1) {
    return "importer failed to verify the IPC pattern";
  }
  return "";
}

// ---------------------------------------------------------------------------
// ROCM-29090 density probe: many processes per GPU exchanging IPC handles.
// ---------------------------------------------------------------------------

// The defect needs this density and the peer imports held open across exports.
constexpr int kDensityProcsPerGpu = 8;
// One process per GPU per group. Fixed rather than probed, because probing it
// needs a HIP call the parent must not make before it forks, and the policy's
// gpu predicate already keeps the test off a box with fewer.
constexpr int kDensityGpus = 4;
constexpr int kDensityProcs = kDensityProcsPerGpu * kDensityGpus;
// Cap; the wall budget below usually binds first.
constexpr int kDensityRounds = 40;
// Buffers each rank exports per round.
constexpr int kDensityBufsPerRank = 11;
// The sizes RCCL exports.
constexpr size_t kDensityBufBytes[] = {2u * 1024 * 1024, 6u * 1024 * 1024};
// A round costs 50 ms on one machine and 900 ms on another, so the round count
// alone cannot bound the run. The budget is what keeps every box inside the
// watchdog; a defective runtime fails in the first second either way.
constexpr long kDensityBudgetMs = 20000;
// How long past the budget the parent waits for the round in flight to drain.
// Wide enough for the first round, where every process brings up the HIP
// runtime and allocates its buffers, several to a GPU; a group that cannot
// finish in it is stuck, and killing it is what keeps the test from hanging.
constexpr long kDensityDrainMs = 30000;
// Grace period before a wedged worker is SIGKILLed.
constexpr long kDensityReapMs = 2000;

// Export latencies kept per process.
constexpr int kDensitySamples = 32;

// Which call a worker's failure came from, so the report can name it.
enum DensityCall {
  kCallNone,
  kCallSetDevice,
  kCallAlloc,
  kCallExport,
  kCallImport,
  kCallClose,
  kCallFree,
};

const char* density_call_name(DensityCall call) {
  switch (call) {
    case kCallSetDevice: return "hipSetDevice";
    case kCallAlloc: return "hipExtMallocWithFlags";
    case kCallExport: return "hipIpcGetMemHandle";
    case kCallImport: return "hipIpcOpenMemHandle";
    case kCallClose: return "hipIpcCloseMemHandle";
    case kCallFree: return "hipFree";
    case kCallNone: break;
  }
  return "";
}

struct DensityStats {
  long exports;
  long imports;
  // The failure that ends the run. hipSuccess and kCallNone are both zero,
  // which is what the shared mapping starts as.
  hipError_t error;
  DensityCall failed_call;
  int samples;
  double sample_us[kDensitySamples];
};

// One failure is enough to end the run, so the round in flight keeps the one it
// started with rather than overwriting it as it winds down.
void density_record_failure(DensityStats* stats, DensityCall call,
                            hipError_t err) {
  if (err == hipSuccess || stats->error != hipSuccess) return;
  stats->error = err;
  stats->failed_call = call;
}

// Cross-process state: the handles, the per-slot flag peers consult before
// importing, the per-group barriers, and one counter block per process.
struct DensityShared {
  std::atomic<int> count[kDensityProcsPerGpu];
  std::atomic<int> generation[kDensityProcsPerGpu];
  // Per group, so the members of one group cannot disagree about the last
  // round: a global flag could flip between two members' reads, leaving one at
  // a barrier its peers had already left.
  std::atomic<int> stop[kDensityProcsPerGpu];
  hipIpcMemHandle_t
      handles[kDensityProcsPerGpu][kDensityGpus][kDensityBufsPerRank];
  std::atomic<int>
      exportable[kDensityProcsPerGpu][kDensityGpus][kDensityBufsPerRank];
  DensityStats stats[kDensityProcs];
  long deadline_ms;  // written by the parent before it forks any worker
  // `src * kDensityGpus + dst + 1` for the first pair that cannot reach each
  // other, 0 while the topology is usable. One field rather than a pair of
  // them, so a reader cannot see half of it.
  std::atomic<int> peer_gap;
};

// These atomics live in the shared mapping.
static_assert(std::atomic<int>::is_always_lock_free,
              "shared atomics must be address-free");

// Every rank checks the whole matrix rather than its own row, so the group
// reaches one verdict and leaves together: a per-row check can pass for some
// ranks and leave the rest waiting at a barrier for peers that gave up.
bool density_peer_access_ok(DensityShared* sh) {
  for (int src = 0; src < kDensityGpus; ++src) {
    for (int dst = 0; dst < kDensityGpus; ++dst) {
      if (src == dst) continue;
      int can_access = 0;
      if (hipDeviceCanAccessPeer(&can_access, src, dst) != hipSuccess ||
          can_access == 0) {
        sh->peer_gap.store(src * kDensityGpus + dst + 1,
                           std::memory_order_release);
        return false;
      }
    }
  }
  return true;
}

// Sense-reversing barrier. No rank may leave early even if a peer never
// arrives: a rank that walked out would export and import against a group that
// is mid-round, and those failures would be counted against the exchange under
// test. A group that cannot make progress is instead killed by the parent's
// bounded reap.
void density_barrier(DensityShared* sh, int group) {
  const int generation = sh->generation[group].load(std::memory_order_acquire);
  if (sh->count[group].fetch_add(1, std::memory_order_acq_rel) ==
      kDensityGpus - 1) {
    sh->count[group].store(0, std::memory_order_release);
    sh->generation[group].fetch_add(1, std::memory_order_acq_rel);
    return;
  }

  while (sh->generation[group].load(std::memory_order_acquire) == generation) {
    sched_yield();
  }
}

// Export this rank's buffers for the round, timing the export. The first
// failure ends the round, leaving the remaining slots unexportable so no peer
// imports a handle this rank never produced.
void density_export_round(DensityShared* sh, DensityStats* stats, int group,
                          int rank, void** exported) {
  // Up front, because a return below must not leave a peer looking at last
  // round's handle, whose memory is gone.
  for (int buf = 0; buf < kDensityBufsPerRank; ++buf) {
    sh->exportable[group][rank][buf].store(0, std::memory_order_relaxed);
    exported[buf] = nullptr;
  }

  for (int buf = 0; buf < kDensityBufsPerRank; ++buf) {
    void* ptr = nullptr;
    hipError_t err = hipExtMallocWithFlags(
        &ptr, kDensityBufBytes[buf % std::size(kDensityBufBytes)],
        hipDeviceMallocUncached);
    if (err != hipSuccess) {
      density_record_failure(stats, kCallAlloc, err);
      return;
    }
    exported[buf] = ptr;

    const auto t0 = PerfClock::now();
    err = hipIpcGetMemHandle(&sh->handles[group][rank][buf], ptr);
    const auto t1 = PerfClock::now();
    if (err != hipSuccess) {
      density_record_failure(stats, kCallExport, err);
      return;
    }

    ++stats->exports;
    if (stats->samples < kDensitySamples) {
      stats->sample_us[stats->samples++] =
          std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    sh->exportable[group][rank][buf].store(1, std::memory_order_relaxed);
  }
}

// Open every peer's exported buffer. The mappings are never read; the density
// the defect needs comes from holding them open, not from their contents.
void density_import_peers(DensityShared* sh, DensityStats* stats, int group,
                          int rank, std::vector<void*>& peers) {
  for (int peer = 0; peer < kDensityGpus; ++peer) {
    if (peer == rank) continue;
    for (int buf = 0; buf < kDensityBufsPerRank; ++buf) {
      // Relaxed is enough for the flag and for the handle bytes beside it: the
      // barrier between the export and this read is what orders them.
      if (sh->exportable[group][peer][buf].load(std::memory_order_relaxed) == 0)
        continue;

      void* remote = nullptr;
      const hipError_t err =
          hipIpcOpenMemHandle(&remote, sh->handles[group][peer][buf],
                              hipIpcMemLazyEnablePeerAccess);
      if (err != hipSuccess) {
        density_record_failure(stats, kCallImport, err);
        return;
      }
      peers.push_back(remote);
      ++stats->imports;
    }
  }
}

// One worker: owns device `rank` and shares it with the other groups.
[[noreturn]] void run_density_worker(DensityShared* sh, int group, int rank) {
  DensityStats* stats = &sh->stats[group * kDensityGpus + rank];
  const hipError_t err = hipSetDevice(rank);
  if (err != hipSuccess) {
    // Cannot enter the barriers: its group would wait for a rank that is gone.
    density_record_failure(stats, kCallSetDevice, err);
    _exit(1);
  }

  // Every import below is cross-device, so without peer access the failures
  // would be topology rather than the defect. Exits clean: the parent reads
  // peer_gap and skips.
  if (!density_peer_access_ok(sh)) _exit(0);

  void* exported[kDensityBufsPerRank] = {};
  std::vector<void*> peers;
  for (int round = 0; round < kDensityRounds; ++round) {
    density_export_round(sh, stats, group, rank, exported);
    density_barrier(sh, group);

    peers.clear();
    density_import_peers(sh, stats, group, rank, peers);
    density_barrier(sh, group);

    for (void* remote : peers) {
      density_record_failure(stats, kCallClose, hipIpcCloseMemHandle(remote));
    }
    // Nobody may free a buffer a peer still holds open.
    density_barrier(sh, group);

    for (int buf = 0; buf < kDensityBufsPerRank; ++buf) {
      if (exported[buf] == nullptr) continue;
      density_record_failure(stats, kCallFree, hipFree(exported[buf]));
      exported[buf] = nullptr;
    }

    // Written only here and read only after the barrier below, so the whole
    // group sees one decision: a rank publishing the moment it failed could
    // land the write after a peer had read and left, and that peer's departure
    // would hang the rest at the next round's first barrier.
    if (stats->error != hipSuccess ||
        posix_util::monotonic_ms() > sh->deadline_ms)
      sh->stop[group].store(1, std::memory_order_release);

    // Holds a rank back from the next round's exports until its peers have
    // finished this one.
    density_barrier(sh, group);
    if (sh->stop[group].load(std::memory_order_acquire) != 0) break;
  }
  _exit(0);
}

// What the run did, which is what makes a failure or its absence quotable.
std::string density_scale(int procs, long exports, long imports) {
  return "procs=" + std::to_string(procs) +
         " gpus=" + std::to_string(kDensityGpus) +
         " exports=" + std::to_string(exports) +
         " imports=" + std::to_string(imports);
}

// `groups` processes per GPU exchange IPC handles for kDensityRounds rounds.
TestResult run_ipc_peer_import_density(int groups, const char* metric) {
  void* shm = ::mmap(nullptr, sizeof(DensityShared), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shm == MAP_FAILED)
    return fail(std::string("mmap(MAP_SHARED|MAP_ANONYMOUS): ") +
                std::strerror(errno));
  auto* sh = new (shm) DensityShared();
  sh->deadline_ms = posix_util::monotonic_ms() + kDensityBudgetMs;

  // fork BEFORE any HIP call so every worker inits the runtime independently.
  pid_t pids[kDensityProcs] = {};
  size_t spawned = 0;
  for (int group = 0; group < groups; ++group) {
    for (int rank = 0; rank < kDensityGpus; ++rank) {
      const pid_t pid = ::fork();
      if (pid < 0) {
        const std::string err = std::strerror(errno);
        (void)posix_util::reap_children_bounded(pids, spawned, kDensityReapMs);
        (void)::munmap(shm, sizeof(DensityShared));
        return fail("fork(worker): " + err);
      }
      if (pid == 0) run_density_worker(sh, group, rank);
      pids[spawned++] = pid;
    }
  }

  // The workers stop themselves on the budget, so this has to outlast it or it
  // would SIGKILL a healthy run mid-flight and report every process as
  // crashed. It is also the only thing bounding a group that cannot progress.
  const int clean_exits = posix_util::reap_children_bounded(
      pids, spawned, kDensityBudgetMs + kDensityDrainMs + kDensityReapMs);
  const int procs = static_cast<int>(spawned);

  long exports = 0;
  long imports = 0;
  hipError_t error = hipSuccess;
  DensityCall failed_call = kCallNone;
  std::vector<double> samples;
  for (int proc = 0; proc < procs; ++proc) {
    const DensityStats& worker = sh->stats[proc];
    exports += worker.exports;
    imports += worker.imports;
    if (error == hipSuccess && worker.error != hipSuccess) {
      error = worker.error;
      failed_call = worker.failed_call;
    }
    samples.insert(samples.end(), worker.sample_us,
                   worker.sample_us + worker.samples);
  }

  const int peer_gap = sh->peer_gap.load(std::memory_order_acquire);
  (void)::munmap(shm, sizeof(DensityShared));

  if (peer_gap != 0)
    return skip("no peer access " + std::to_string((peer_gap - 1) /
                                                   kDensityGpus) +
                " -> " + std::to_string((peer_gap - 1) % kDensityGpus) +
                "; every import here is cross-device");

  const std::string scale = density_scale(procs, exports, imports);
  const std::string failure =
      error == hipSuccess ? std::string()
                          : std::string(density_call_name(failed_call)) + ": " +
                                hipGetErrorString(error);

  // A run that exported nothing or lost a process proves nothing, so it must
  // not read as a pass. Its failure, if it had one, is usually why.
  if (exports == 0 || clean_exits != procs)
    return fail("inconclusive: " + scale +
                " crashed=" + std::to_string(procs - clean_exits) +
                (failure.empty() ? "" : " (" + failure + ")"));

  // A defective runtime never reaches the pass path, so the latencies have to
  // be emitted before it.
  PerfSink sink;
  emit_latency_samples(sink, metric, std::move(samples));

  if (!failure.empty()) {
    TestResult result = fail(failure + " (" + scale + ")");
    result.perf = std::move(sink);
    return result;
  }
  return pass_with_perf(std::move(sink));
}

}  // namespace

// Forked parent exports an IPC mem handle; child imports it and kernel-verifies the pattern.
ACCEPT_TEST(Ipc_Legacy_GetOpen_RoundTrip, "hip/ipc", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = 1024;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  const int kWaitMs = 15000;  // bounded; runner watchdog is the hard backstop

  // Shared region for sync flags + the IPC handle.
  void* shm = ::mmap(nullptr, sizeof(IpcShared), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shm == MAP_FAILED)
    return fail(std::string("mmap(MAP_SHARED|MAP_ANONYMOUS): ") +
                std::strerror(errno));
  auto* sh = static_cast<IpcShared*>(shm);
  sh->handle_ready = 0;
  sh->child_done = 0;
  sh->child_ok = 0;

  // fork BEFORE any HIP call so both processes init HIP independently.
  const pid_t pid = ::fork();
  if (pid < 0) {
    (void)::munmap(shm, sizeof(IpcShared));
    return fail(std::string("fork: ") + std::strerror(errno));
  }

  if (pid == 0) {
    run_ipc_importer_child(sh, n, kWaitMs);
  }

  // Parent: buf/stream stay owned here so they free on scope exit, after the
  // bounded reap+unmap tail below.
  DeviceBuffer buf;
  Stream stream;
  PerfSink perf_sink;
  const std::string err =
      run_ipc_exporter_parent(sh, buf, stream, n, bytes, kWaitMs, perf_sink);

  // Bounded reap: a wedged importer is SIGKILLed rather than blocking the parent
  // forever on waitpid — preserves the no-hang guarantee.
  const int wstatus = posix_util::reap_child_bounded(pid, kReapGraceMs);
  (void)::munmap(shm, sizeof(IpcShared));

  if (!err.empty())
    return fail(err + " (child exit status " + std::to_string(wstatus) + ")");

  return pass_with_perf(std::move(perf_sink));
}

// Cross-device IPC handle exchange at one process per GPU: the density control
// for Ipc_Export_PeerImports_MultiProcess, and the only coverage of peer
// imports between two devices (the round-trip test above stays on device 0).
ACCEPT_TEST(Ipc_Export_PeerImports_OneProcess, "hip/ipc", Topology::Single,
            Kind::Functional, 120) {
  (void)ctx;
  return run_ipc_peer_import_density(1, "ipc.get_handle_solo");
}

// ROCM-29090: the same exchange with kDensityProcsPerGpu processes per GPU,
// where the export intermittently fails.
ACCEPT_TEST(Ipc_Export_PeerImports_MultiProcess, "hip/ipc", Topology::Single,
            Kind::Functional, 120) {
  (void)ctx;
  return run_ipc_peer_import_density(kDensityProcsPerGpu,
                                     "ipc.get_handle_dense");
}
}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
