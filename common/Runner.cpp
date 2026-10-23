/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/Runner.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>

#include "common/Bootstrap.hpp"
#include "common/Policy.hpp"
#include "common/PosixUtil.hpp"
#include "common/Reporter.hpp"
#include "common/Perf.hpp"
#include "common/TestResult.hpp"
#include "common/Types.hpp"
#include "wrapper/hip/api_wrapper.hpp"
#ifdef ACCEPT_HAVE_IBVERBS
#include "wrapper/verbs/ibv_api_wrapper.hpp"
#endif

namespace acceptance {

namespace {

// ---------------------------------------------------------------------------
// Small POSIX helpers
// ---------------------------------------------------------------------------

// waitpid_eintr / sleep_ms live in common/PosixUtil.hpp (shared).
using namespace posix_util;

// pid (== process group) of the isolated child run_isolated() is waiting on,
// or 0. Read by kill_active_isolated_test(), including from signal handlers.
std::atomic<pid_t> g_active_isolated_pid{0};
static_assert(std::atomic<pid_t>::is_always_lock_free);

// Clears g_active_isolated_pid when run_isolated() returns, on every path.
struct ActiveIsolatedPid {
  explicit ActiveIsolatedPid(pid_t pid) { g_active_isolated_pid.store(pid); }
  ~ActiveIsolatedPid() { g_active_isolated_pid.store(0); }
  ActiveIsolatedPid(const ActiveIsolatedPid&) = delete;
  ActiveIsolatedPid& operator=(const ActiveIsolatedPid&) = delete;
};

// Human-readable signal name (best effort, portable subset).
std::string signal_name(int sig) {
#if defined(__GLIBC__) || defined(__APPLE__)
  const char* n = strsignal(sig);
  if (n != nullptr) return n;
#endif
  return "signal " + std::to_string(sig);
}

double duration_ms_since(const std::chrono::steady_clock::time_point& t0) {
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Build a synthesized Fail row. `synth` is "crash" | "timeout" | "corrupt".
TestResult synth_fail(const std::string& id, const char* synth,
                      std::string reason, int exit_code, double dur_ms) {
  TestResult r;
  r.id = id;
  r.status = Status::Fail;
  r.reason = std::move(reason);
  r.duration_ms = dur_ms;
  r.exit_code = exit_code;
  r.synth = synth;
  return r;
}

// Read & parse the child's result file. On any problem (missing / unparseable /
// wrong id) returns false and fills `out_synth_reason`; on success returns true
// with `out` populated. The child's exit code (already known to be 0) is the
// authority that we even bother reading the file.
bool read_child_result(const std::string& path, const std::string& expect_id,
                       TestResult& out, std::string& out_synth_reason) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    out_synth_reason = "result file missing";
    return false;
  }
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  if (text.empty()) {
    out_synth_reason = "result file missing";  // empty == nothing written
    return false;
  }

  // key=value parse (tolerant, never throws). A garbage file simply yields an
  // empty/default id, which fails the id check below -> "corrupt" (§6.3).
  {
    std::stringstream is(text);
    out = TestResult::from_kv(is);
  }

  if (out.id != expect_id) {
    out_synth_reason = "wrong test id in result";
    return false;
  }
  return true;
}

}  // namespace

bool Runner::should_isolate(const TestMeta& meta) const {
  // TwoNode tests never isolate (forked child cannot share the live socket).
  if (meta.topology == Topology::TwoNode) return false;
  if (opts_.isolate == "none") {
    // Debug scope: run in-process. Some tests MUST still isolate even here,
    // because running them in the shared worker corrupts it: hip/ipc forks and
    // re-inits HIP in the child (a fork from a worker that already initialized
    // HIP corrupts the child's HSA state -> the importer fails); hip/devreset
    // calls hipDeviceReset which tears down the shared worker; vat_fork_init
    // calls ibv_fork_init, a process-global switch that must precede ALL MR
    // registration -- a fresh process guarantees it runs first and does not
    // taint sibling verbs tests; the Vmm_Reclaim_Cycles_* probes hold their VA
    // reservations by design, so only process exit gives the memory back.
    if (meta.category == "hip/ipc" || meta.category == "hip/devreset")
      return true;
    if (meta.id == "vat_fork_init") return true;
    if (meta.id == "Vmm_Reclaim_Cycles_InProcess") return true;
    if (meta.id == "Vmm_Reclaim_Cycles_CrossProcess") return true;
    if (meta.id == "Launch_Cooperative_AfterQueueReclaim") return true;
    return false;
  }
  // "all" (default) and any unrecognized scope: isolate every test.
  return true;
}

// In-process path has no watchdog; crash/hang tests are always isolated.
TestResult Runner::run_in_process(const TestMeta& meta) const {
  // Use the shared bootstrap (e.g. SocketBootstrap for a 2-rank worker) when one
  // was supplied; otherwise fall back to a per-test loopback (single-rank).
  LoopbackBootstrap loopback;
  TestContext ctx;
  if (opts_.boot != nullptr) {
    ctx.boot = opts_.boot;
    ctx.rank = opts_.rank;
    ctx.size = opts_.size;
  } else {
    ctx.boot = &loopback;
    ctx.rank = loopback.rank();
    ctx.size = loopback.size();
  }

  const auto t0 = std::chrono::steady_clock::now();
  TestResult result = run_with_probe([&] {
    try {
      return meta.fn(ctx);
    } catch (const hipw::MissingSymbol& e) {
      return skip(std::string("requires ") + e.name);
    }
  });
  result.duration_ms = duration_ms_since(t0);
  result.id = meta.id;  // body never sets id; runner is authoritative
  return result;
}

// Isolated: fork + execv --exec; parent polls waitpid (EINTR-safe, no zombies).
TestResult Runner::run_isolated(const TestMeta& meta) const {
  const auto t0 = std::chrono::steady_clock::now();

  // Unique temp file for the child's result, one per test.
  char tmpl[] = "/tmp/accept_result_XXXXXX";
  const int fd = ::mkstemp(tmpl);
  if (fd < 0) {
    // Harness-level failure: with no result file the child would (per do_exec)
    // write its result to STDOUT and exit 0 — leaking output into the parent and
    // making us synthesize a misleading "corrupt". Bail out cleanly before fork.
    return synth_fail(meta.id, "crash",
                      std::string("mkstemp failed: ") + std::strerror(errno),
                      -1, duration_ms_since(t0));
  }
  const std::string result_path = tmpl;
  ::close(fd);  // child reopens by path; we only needed a unique name

  const pid_t parent = ::getpid();
  const pid_t pid = ::fork();
  if (pid < 0) {
    if (!result_path.empty()) ::unlink(result_path.c_str());
    return synth_fail(meta.id, "crash", "fork failed", -1, duration_ms_since(t0));
  }

  if (pid == 0) {
    // Process group so kill(-pid) reaches grandchildren on timeout.
    ::setpgid(0, 0);
#ifdef __linux__
    // Die with this process. If it is killed (worker deadline, launcher
    // signal, OOM), nothing else would enforce this child's timeout.
    // Survives the execv below.
    ::prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (::getppid() != parent) ::_exit(127);  // parent died before prctl
#endif
    // Re-exec self in EXEC mode. /proc/self/exe is the running binary.
    // TEST HOOK: syn_corrupt_result needs an exit-0 child that nonetheless
    // leaves a corrupt result file. We set ACCEPT_SYN_CORRUPT=1 in the child
    // env; do_exec() in main.cc honors it by writing garbage then exiting 0.
    if (meta.id == "syn_corrupt_result") {
      ::setenv("ACCEPT_SYN_CORRUPT", "1", 1);
    }
    char self[] = "/proc/self/exe";
    char arg_exec[] = "--exec";
    char arg_rf[] = "--result-file";
    char arg_kind[] = "--kind";
    char arg_perf[] = "perf";
    // execv needs mutable C strings; copy id and path into local buffers.
    std::string id_copy = meta.id;
    std::string rf_copy = result_path;
    // The isolated child re-parses args from scratch, so forward perf mode
    // explicitly; otherwise it defaults to --kind func (measurement off).
    std::vector<char*> argv = {self, arg_exec, id_copy.data(),
                               arg_rf, rf_copy.data()};
    if (perf_probe_mode()) { argv.push_back(arg_kind); argv.push_back(arg_perf); }
    argv.push_back(nullptr);
    ::execv(self, argv.data());
    // execv only returns on failure. Use _exit (not exit) to avoid flushing
    // inherited stdio buffers from the parent. Exit code 127 = exec failure;
    // the parent will read this as a non-zero exit -> synth crash.
    ::_exit(127);
  }

  ::setpgid(pid, pid);  // parent-side race close; errors ignored
  const ActiveIsolatedPid active(pid);

  const long timeout_ms =
      static_cast<long>(meta.timeout_sec) * 1000L;
  const long poll_ms = 20;  // poll cadence
  long waited_ms = 0;
  int status = 0;
  bool reaped = false;
  bool timed_out = false;

  for (;;) {
    const pid_t r = waitpid_eintr(pid, &status, WNOHANG);
    if (r == pid) {
      reaped = true;
      break;
    }
    if (r < 0) {
      // Unexpected (e.g. ECHILD): treat as crash; nothing left to reap.
      const int werr = errno;  // preserve waitpid's errno across unlink()
      reaped = true;           // can't reap something that's gone
      if (!result_path.empty()) ::unlink(result_path.c_str());
      return synth_fail(meta.id, "crash",
                        std::string("waitpid failed: ") + std::strerror(werr),
                        -1, duration_ms_since(t0));
    }
    // r == 0: still running.
    if (waited_ms >= timeout_ms) {
      timed_out = true;
      break;
    }
    sleep_ms(poll_ms);
    waited_ms += poll_ms;
  }

  if (timed_out) {
    // Escalate: SIGTERM, grace period, then SIGKILL. Reap after each stage.
    //
    // Signal the WHOLE process group (negative pid == the group whose leader is
    // `pid`, established via setpgid above) so any grandchildren the test
    // fork()'d are taken down with the direct child — not just the EXEC child.
    // We still waitpid(pid) (positive, the direct child) to REAP it: only the
    // direct child is ours to reap. Grandchildren are SIGKILL'd here and then
    // reaped by init (PID 1) after we exit their group — we never waitpid() for
    // them, so they are never zombies *for us* (no ECHILD spin, no leak).
    ::kill(-pid, SIGTERM);
    const long grace_ms = static_cast<long>(opts_.kill_grace_sec) * 1000L;
    long g = 0;
    while (g < grace_ms) {
      const pid_t r = waitpid_eintr(pid, &status, WNOHANG);
      if (r == pid) {
        reaped = true;
        break;
      }
      sleep_ms(poll_ms);
      g += poll_ms;
    }
    if (!reaped) {
      ::kill(-pid, SIGKILL);
      // Blocking reap of the DIRECT child — SIGKILL cannot be caught, it dies
      // promptly. Grandchildren also received SIGKILL (group signal) and are
      // reaped by init, not us.
      waitpid_eintr(pid, &status, 0);
      reaped = true;
    }
    if (!result_path.empty()) ::unlink(result_path.c_str());
    // exit_code reflects how the child ACTUALLY terminated, so the artifact can
    // distinguish a well-behaved test (honored SIGTERM in the grace window ->
    // -SIGTERM, or a clean exit code) from one that had to be force-killed
    // (-SIGKILL). `status` was captured by the reaping waitpid above. The
    // "timeout" synth marker remains authoritative for the gate regardless.
    int exit_code = -SIGKILL;  // fallback if no status was captured
    if (WIFSIGNALED(status))      exit_code = -WTERMSIG(status);
    else if (WIFEXITED(status))   exit_code = WEXITSTATUS(status);
    return synth_fail(meta.id, "timeout",
                      "timeout after " + std::to_string(meta.timeout_sec) + "s",
                      exit_code, duration_ms_since(t0));
  }

  (void)reaped;  // child reaped above on the normal path

  // Child exited within its timeout. Exit status is authoritative.
  const double dur = duration_ms_since(t0);

  if (WIFSIGNALED(status)) {
    const int sig = WTERMSIG(status);
    if (!result_path.empty()) ::unlink(result_path.c_str());
    return synth_fail(meta.id, "crash",
                      "killed by signal " + std::to_string(sig) + " (" +
                          signal_name(sig) + ")",
                      -sig, dur);
  }

  if (WIFEXITED(status)) {
    const int code = WEXITSTATUS(status);
    if (code != 0) {
      // Non-zero exit is AUTHORITATIVE — do not trust any file content.
      if (!result_path.empty()) ::unlink(result_path.c_str());
      return synth_fail(meta.id, "crash",
                        "exited with code " + std::to_string(code), code, dur);
    }
    // Exit 0: read & validate the child's result file.
    TestResult parsed;
    std::string why;
    const bool ok =
        result_path.empty()
            ? (why = "result file missing", false)
            : read_child_result(result_path, meta.id, parsed, why);
    if (!result_path.empty()) ::unlink(result_path.c_str());
    if (!ok) {
      return synth_fail(meta.id, "corrupt", std::move(why), 0, dur);
    }
    // Trust the child's row. Ensure id + a duration are set.
    parsed.id = meta.id;
    if (parsed.duration_ms == 0) parsed.duration_ms = dur;
    parsed.exit_code = 0;
    // synth stays whatever the child wrote (normally ""); a real test never
    // sets it, so it is empty for genuine pass/fail/skip.
    return parsed;
  }

  // WIFSTOPPED/WIFCONTINUED should not occur (we don't WUNTRACED); be safe.
  if (!result_path.empty()) ::unlink(result_path.c_str());
  return synth_fail(meta.id, "crash", "child in unexpected wait state", -1, dur);
}

// Stream one finished test's GTest-style line to stderr (flushed). For a single
// rank, reduce({r.status}) == r.status, so the streamed tag equals the final
// tag the Reporter will emit. We reuse the shared formatter so the live line is
// byte-identical to the Reporter end-block line.
void Runner::stream_result(const TestResult& r) const {
  const Status reduced = reduce({r.status});
  Classification c;  // defaults to Class::Unclassified
  auto it = opts_.classifications.find(r.id);
  if (it != opts_.classifications.end()) c = it->second;
  const Verdict v = apply_classification(reduced, c);
  const std::string detail = pick_detail(v, r.synth, r.reason, c.reason);
  const std::string line = format_test_line(r.id, v.status, detail);
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

std::vector<TestResult> Runner::run(const std::vector<TestMeta>& tests) const {
  std::vector<TestResult> out;
  out.reserve(tests.size());
  std::string last_category;

  // INVARIANT: opts_.skipped_lines must be ordered by category in the same
  // sequence as TestRegistry::all() (which is stable-sorted by category).
  // classify_and_partition() builds skipped_lines by iterating the same sorted
  // registry, so this is guaranteed as long as both iterate the same source.
  // If that ever changes, a sort here would be needed.
  std::size_t skip_idx = 0;

  auto flush_skipped_for = [&](const std::string& category) {
    while (skip_idx < opts_.skipped_lines.size() &&
           opts_.skipped_lines[skip_idx].first == category) {
      std::fprintf(stderr, "%s\n", opts_.skipped_lines[skip_idx].second.c_str());
      std::fflush(stderr);
      ++skip_idx;
    }
  };

  for (const auto& meta : tests) {
    if (opts_.stream) {
      const std::string hdr = format_group_header(meta.category, last_category);
      if (!hdr.empty()) {
        std::fprintf(stderr, "%s\n", hdr.c_str());
        std::fflush(stderr);
        // Flush any NA/SKIP lines belonging to this category before its tests.
        flush_skipped_for(meta.category);
      }
    }
    TestResult r;
    bool category_gated = false;
    if (hipw::hip_category_unavailable(meta.category)) {
      // No ROCm on this host: skip the whole HIP category up front (one clean
      // reason) rather than fork/exec a child that would only skip anyway.
      r = skip("HIP runtime unavailable (libamdhip64.so not loaded)");
      category_gated = true;
    }
#ifdef ACCEPT_HAVE_IBVERBS
    else if (ibvw::verbs_category_unavailable(meta.category)) {
      // No RDMA stack on this host: skip the whole verbs category up front.
      r = skip("verbs runtime unavailable (libibverbs.so.1 not loaded)");
      category_gated = true;
    }
#endif
    if (category_gated) {
      r.id = meta.id;
    } else {
      r = should_isolate(meta) ? run_isolated(meta) : run_in_process(meta);
    }
    if (opts_.stream) stream_result(r);
    out.push_back(std::move(r));
  }

  // Flush any remaining skipped lines for categories with no runnable tests.
  if (opts_.stream) {
    while (skip_idx < opts_.skipped_lines.size()) {
      const std::string& cat = opts_.skipped_lines[skip_idx].first;
      const std::string hdr = format_group_header(cat, last_category);
      if (!hdr.empty()) {
        std::fprintf(stderr, "%s\n", hdr.c_str());
        std::fflush(stderr);
      }
      std::fprintf(stderr, "%s\n", opts_.skipped_lines[skip_idx].second.c_str());
      std::fflush(stderr);
      ++skip_idx;
    }
  }

  return out;
}

void kill_active_isolated_test(int sig) {
  const pid_t pid = g_active_isolated_pid.load();
  if (pid > 0) ::kill(-pid, sig);
}

}  // namespace acceptance
