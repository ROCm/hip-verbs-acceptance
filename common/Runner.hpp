/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Runner — the WORKER core. Given a selection of TestMeta and run options, it
// produces exactly one TestResult per test. Each test runs either in-process
// (trivially-safe tests) or process-isolated (fork + exec self in --exec mode)
// under a per-test watchdog that contains crashes, hangs, and corrupt output.
//
// The isolated path is the safety-critical code: exit status is AUTHORITATIVE
// over the child's result file (ARCHITECTURE.md §6.3), and any
// missing/unparseable/wrong-id file from an exit-0 child is synthesized as a
// "corrupt" Fail. Crashes (signal or non-zero exit) and timeouts are likewise
// synthesized into Fail rows so every test yields one uniform result schema.

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/Bootstrap.hpp"
#include "common/Policy.hpp"
#include "common/TestRegistry.hpp"
#include "common/TestResult.hpp"
#include "common/Types.hpp"

namespace acceptance {

struct RunnerOptions {
  // Isolation scope:
  //   "all" (default): isolate every test in a forked+exec'd watchdog child, so
  //                    a crash/hang/timeout in any test can never take down the
  //                    run. TwoNode tests are the one exception (a forked child
  //                    cannot share the live rendezvous socket) and run
  //                    in-process regardless.
  //   "none": run every test in-process (NO watchdog; a crash/hang takes down the
  //           run). For single-test debugging only — the launcher selects this
  //           automatically when exactly one test runs and --isolate is implicit.
  std::string isolate = "all";
  // Pre-formatted NA/SKIP lines (category + text) to interleave with the live
  // stream. The Runner flushes them at the start of each category group so they
  // appear under the correct header, before the runnable tests for that group.
  std::vector<std::pair<std::string, std::string>> skipped_lines;  // {category, line}

  // Grace period between SIGTERM and SIGKILL when a child overruns its timeout.
  int kill_grace_sec = 2;

  // Shared bootstrap for IN-PROCESS tests (Step 5). When non-null, in-process
  // tests (e.g. cooperative TwoNode tests) receive this bootstrap and rank/size
  // in their TestContext instead of a fresh LoopbackBootstrap. Isolated (forked)
  // tests can never use it — the live socket does not survive exec — so TwoNode
  // tests must be run non-isolated (the worker arranges this). Null => the Runner
  // falls back to a per-test LoopbackBootstrap (single-rank behavior).
  Bootstrap* boot = nullptr;
  int        rank = 0;
  int        size = 1;

  // Live streaming (additive console output). When `stream` is true, the Runner
  // prints one GTest-style per-test line to STDERR (flushed immediately) the
  // moment each test finishes, using the shared Reporter formatter. This makes
  // results appear incrementally during the run instead of all at the end. The
  // returned vector is unchanged — streaming is purely extra console output.
  bool stream = false;

  // Discovery mode (no policy in effect). Passed through to
  // apply_classification() when computing the live verdict so the streamed tag
  // matches what the Reporter will later emit.
  bool discovery = false;

  // Per-test full Classification, keyed by test id. Used ONLY for the live
  // stream's verdict/detail (the gate itself is still computed by the Reporter).
  // A test id absent from the map is treated as Class::Unclassified.
  std::map<std::string, Classification> classifications;

  // Called before each in-process test body (e.g. to make the selected GPU
  // current again after the previous test); a non-empty return fails the test
  // with that reason without running it. Isolated children do their own.
  std::function<std::string(const TestMeta&)> prepare;
};

class Runner {
 public:
  explicit Runner(RunnerOptions opts) : opts_(std::move(opts)) {}

  // Run every test in `tests`, in order, returning one TestResult per test
  // (same order). Never throws for test-level failures — those become Fail
  // rows. The vector is always tests.size() long.
  std::vector<TestResult> run(const std::vector<TestMeta>& tests) const;

  // Whether a given test would be isolated under the current options. Exposed
  // for testing / introspection.
  bool should_isolate(const TestMeta& meta) const;

 private:
  // Run one test in the current process (no watchdog — see ARCHITECTURE note).
  TestResult run_in_process(const TestMeta& meta) const;

  // Run one test in a forked+exec'd child under the watchdog.
  TestResult run_isolated(const TestMeta& meta) const;

  // Stream one finished test's GTest-style line to stderr (flushed). No-op
  // unless opts_.stream is set; called by run() after each test completes.
  void stream_result(const TestResult& r) const;

  RunnerOptions opts_;
};

// Send `sig` to the process group of the isolated test child this process is
// currently waiting on, if any. That child leads its own group, so killing the
// worker's group does not reach it. Async-signal-safe (an atomic load and
// kill()), for the worker's deadline watchdog and its SIGTERM/SIGINT handler.
void kill_active_isolated_test(int sig);

}  // namespace acceptance
