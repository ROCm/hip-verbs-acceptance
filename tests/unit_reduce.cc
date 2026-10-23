/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_reduce — deterministic unit test for the pure Reporter functions
// reduce() (the §7.3 reduction lattice) and apply_classification() (the §7.4
// classification/gating table). No test framework: a hand-rolled CHECK macro
// counts failures and the process exits non-zero if any assertion fails.
//
// These functions are the heart of the gate verdict and must stay correct
// independent of the synthetic fixtures / report plumbing. The reduce() cases
// already exercise multi-rank vectors so Step 5 (2-rank reconciliation) inherits
// coverage of None=identity and the fail>skip>pass ordering.

#include <cstdio>
#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/Policy.hpp"
#include "common/Reporter.hpp"
#include "common/Types.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

// Shared counters for the whole unit_tests binary (declared extern in
// unit_harness.hpp; the other unit TUs bump these via CHECK).
namespace accept_test {
int g_failures = 0;
int g_checks = 0;
}  // namespace accept_test

namespace {

// This TU keeps a local CHECK that targets the shared counters (so the existing
// reduce/classification cases below are unchanged in spirit). New TUs use the
// CHECK from unit_harness.hpp directly.
#undef CHECK
#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    ++::accept_test::g_checks;                                              \
    if (!(cond)) {                                                          \
      ++::accept_test::g_failures;                                          \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                       \
  } while (0)

const char* st(Status s) { return to_string(s); }
const char* fst(FinalStatus s) { return to_string(s); }

// reduce() helper that reports the actual vs expected on mismatch.
void check_reduce(const std::vector<Status>& in, Status want, const char* msg) {
  Status got = reduce(in);
  ++accept_test::g_checks;
  if (got != want) {
    ++accept_test::g_failures;
    std::fprintf(stderr, "FAIL reduce %s: got=%s want=%s\n", msg, st(got),
                 st(want));
  }
}

// apply_classification() helper. Asserts status / gate_fail / flagged together.
void check_verdict(Status reduced, const Classification& c,
                   FinalStatus want_status, bool want_gate, bool want_flag,
                   const char* msg) {
  Verdict v = apply_classification(reduced, c);
  ++accept_test::g_checks;
  if (v.status != want_status || v.gate_fail != want_gate ||
      v.flagged != want_flag) {
    ++accept_test::g_failures;
    std::fprintf(stderr,
                 "FAIL verdict %s: got=(%s,gate=%d,flag=%d) "
                 "want=(%s,gate=%d,flag=%d)\n",
                 msg, fst(v.status), v.gate_fail, v.flagged, fst(want_status),
                 want_gate, want_flag);
  }
}

Classification cls(Class c, std::string expected = "") {
  Classification x;
  x.cls = c;
  x.expected_status = std::move(expected);
  return x;
}

void test_reduce() {
  // §7.3 lattice: drop None; any Fail -> Fail; else any Skip -> Skip; else Pass;
  // empty / all-None -> None.
  check_reduce({Status::Fail, Status::Pass}, Status::Fail, "[Fail,Pass]");
  check_reduce({Status::Skip, Status::Pass}, Status::Skip, "[Skip,Pass]");
  check_reduce({Status::None, Status::Pass}, Status::Pass, "[None,Pass]");
  check_reduce({Status::Pass, Status::Pass}, Status::Pass, "[Pass,Pass]");
  check_reduce({Status::None, Status::None}, Status::None, "[None,None]");
  check_reduce({}, Status::None, "[]");
  check_reduce({Status::Fail, Status::Skip, Status::Pass, Status::None},
               Status::Fail, "[Fail,Skip,Pass,None]");
  check_reduce({Status::Skip, Status::None}, Status::Skip, "[Skip,None]");
}

void test_apply_classification() {
  // MUST_PASS (§7.4): Pass->pass(no gate); Fail->fail(gate); Skip->fail(gate);
  // None->na(no gate).
  check_verdict(Status::Pass, cls(Class::MustPass), FinalStatus::Pass,
                false, false, "MUST_PASS Pass");
  check_verdict(Status::Fail, cls(Class::MustPass), FinalStatus::Fail,
                true, false, "MUST_PASS Fail");
  check_verdict(Status::Skip, cls(Class::MustPass), FinalStatus::Fail,
                true, false, "MUST_PASS Skip");
  check_verdict(Status::None, cls(Class::MustPass), FinalStatus::Na,
                false, false, "MUST_PASS None");

  // KNOWN_FAILURE: always outputs Fail to create fix pressure.
  // Fail matches expected -> note "known failure"; mismatch -> note "possibly fixed".
  // None -> Na (test didn't run).
  check_verdict(Status::Fail, cls(Class::KnownFailure, "fail"),
                FinalStatus::Fail, true, false, "KNOWN_FAILURE Fail (expected)");
  check_verdict(Status::Pass, cls(Class::KnownFailure, "fail"),
                FinalStatus::Fail, true, false, "KNOWN_FAILURE Pass (unexpected)");
  // None -> Na: the test did not run on any rank (e.g. all workers were passive
  // for a TwoNode test on a single-rank run). Gate must NOT fire in this case.
  check_verdict(Status::None, cls(Class::KnownFailure, "fail"),
                FinalStatus::Na, false, false, "KNOWN_FAILURE None -> Na (gate-neutral)");

  // SKIP: software-gated; gate-neutral.
  check_verdict(Status::None, cls(Class::Skip),
                FinalStatus::Skip, false, false, "SKIP -> skip, no gate");

  // NA: hardware-permanent; gate-neutral.
  check_verdict(Status::None, cls(Class::Na),
                FinalStatus::Na, false, false, "NA None");
}

void test_pick_predicate_env() {
  using NodeInfo = ReportInputs::NodeInfo;

  // Empty -> static empty env (no rocm).
  {
    std::vector<NodeInfo> nodes;
    const EnvironmentInfo& e = pick_predicate_env(nodes);
    CHECK(e.rocm_version.empty(), "pick_predicate_env: empty nodes -> empty env");
  }

  // Single node -> its env.
  {
    std::vector<NodeInfo> nodes;
    NodeInfo n;
    n.rank = 0;
    n.environment.rocm_version = "7.0.1";
    nodes.push_back(n);
    const EnvironmentInfo& e = pick_predicate_env(nodes);
    CHECK_EQ_STR(e.rocm_version, "7.0.1", "pick_predicate_env: single node");
  }

  // Determinism: nodes in REVERSE rank order must still pick rank 0's env (the
  // bug this guards is relying on nodes[0]/insertion order).
  {
    std::vector<NodeInfo> nodes;
    NodeInfo n1;
    n1.rank = 1;
    n1.environment.rocm_version = "9.9-rank1";
    nodes.push_back(n1);  // rank 1 inserted FIRST
    NodeInfo n0;
    n0.rank = 0;
    n0.environment.rocm_version = "7.0-rank0";
    nodes.push_back(n0);  // rank 0 second
    const EnvironmentInfo& e = pick_predicate_env(nodes);
    CHECK_EQ_STR(e.rocm_version, "7.0-rank0",
                 "pick_predicate_env: lowest rank wins regardless of order");
  }
}

}  // namespace

int main() {
  // Reporter reduce/classification (this TU).
  test_reduce();
  test_apply_classification();
  test_pick_predicate_env();
  // Policy CSV parse/classify/load, ResultIO/TestResult round-trip, ssh/CLI
  // builders + quoting, rank placement + worker argv, GPU/HCA selection, perf key grammar/rank merge/rendering, Capability's
  // pure HIP-free helpers (sibling TUs, sharing the counters).
  accept_test::run_policy_tests();
  accept_test::run_resultio_tests();
  accept_test::run_ssh_tests();
  accept_test::run_launch_tests();
  accept_test::run_topology_tests();
  accept_test::run_perf_tests();
  accept_test::run_capability_tests();

  std::printf("unit_tests: %d checks, %d failures\n", accept_test::g_checks,
              accept_test::g_failures);
  if (accept_test::g_failures != 0) {
    std::printf("UNIT TESTS FAILED\n");
    return 1;
  }
  std::printf("UNIT TESTS PASSED\n");
  return 0;
}
