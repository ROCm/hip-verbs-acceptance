/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Reporter — turns raw per-rank worker results into a classified report.json with
// gate verdicts (ARCHITECTURE.md §7.3/§7.4, §8). Three concerns:
//   1. reduce()              — collapse per-rank Status into one (the §7.3 lattice)
//   2. apply_classification() — map (reduced, Classification) -> Verdict (§7.4)
//   3. assemble + write the report.json document, print a human summary, and
//      return the overall gate exit code (0 = gate pass, 1 = any gate_fail).
//
// NA synthesis: NA-classified tests are never run by the worker. The Reporter
// synthesizes their rows from the SELECTED test set x policy x vendor (§7.1/§8),
// so the report covers every selected test, run or not.
//
// Selection scoping: the Reporter emits rows ONLY for the selected test set (the
// same set the worker ran), never the whole registry. This keeps a --filter/--kind
// run from silently turning EXCLUDED tests into synthesized rows (which would let
// an omitted MUST_PASS failure masquerade as `na` and pass the gate). For a full
// unfiltered run selected == the whole registry, so behavior is unchanged.

#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/PerfSummary.hpp"
#include "common/Policy.hpp"
#include "common/TestRegistry.hpp"
#include "common/TestResult.hpp"
#include "common/Types.hpp"

namespace acceptance {

// Reduce per-rank statuses for ONE test (the §7.3 lattice):
//   drop None; if none remain -> None (means "all passive / not participating");
//   else any Fail -> Fail; else any Skip -> Skip; else Pass.
Status reduce(const std::vector<Status>& per_rank);

// Outcome of applying a Class to a reduced Status.
struct Verdict {
  FinalStatus status = FinalStatus::Pass;
  bool        gate_fail = false;  // true => fails the acceptance gate
  bool        flagged = false;    // true => needs human attention (loud)
  std::string note;               // e.g. "unclassified", "possibly fixed"
};

// Apply classification to a reduced status (§7.4).
Verdict apply_classification(Status reduced, const Classification& c);

// ---------------------------------------------------------------------------
// Shared per-test line formatter (used by BOTH the Reporter end-block and the
// Runner's live stream, so the two produce byte-identical lines).
// ---------------------------------------------------------------------------

// GTest-style fixed-width status tag: " PASS  ", " FAIL  ", " SKIP  ", "  NA   ",
// "KN-FAIL", " UNCLS ".
const char* status_tag(FinalStatus s);

// Format one GTest-style line: "[ <tag> ] <id>  (<detail>)" — the "(<detail>)"
// suffix is omitted when `detail` is empty.
std::string format_test_line(const std::string& id, FinalStatus status,
                             const std::string& detail);

// Format a suite group header: "\n--- <category> ---".
// Returns an empty string when `category` equals `last_category` (no change),
// and updates `last_category` to `category` when a header is produced.
// Callers keep one `last_category` variable across a sequence of test lines.
std::string format_group_header(const std::string& category,
                                std::string& last_category);

// Pick the detail string by the documented precedence: verdict note, else the
// synthesized-failure marker, else the per-rank reason, else the policy reason.
// The policy reason is suppressed when the verdict is Pass (it explains a
// non-run/known outcome, not a pass), so a passing test shows no policy detail.
std::string pick_detail(const Verdict& v, const std::string& synth,
                        const std::string& rank_reason,
                        const std::string& cls_reason);

// Inputs for assembling the report.
struct ReportInputs {
  std::string                              vendor;
  bool                                     discovery = false;  // policy absent
  // The SELECTED test ids (post --filter/--kind). The Reporter emits a row for
  // each of these and nothing else. NA synthesis is limited to this set too.
  std::set<std::string>                    selected_ids;
  // --filter-class <class>, empty = no class filter. Applied after
  // classification (with the predicate env), the same way the worker's
  // classify_and_partition() applies it, so a test the class filter excluded
  // from the run is also excluded from the report instead of being counted
  // as a missing (structural-failure) result.
  std::string                              class_filter;
  // true when the selection is narrower than the full registry (a partial report,
  // not a full acceptance gate). Surfaced as report.json top-level `filtered`.
  bool                                     filtered = false;
  // When false (default), print one GTest-style line per test to stdout before
  // the summary. --quiet sets this true to print only the summary line (and
  // suppresses the performance block).
  bool                                     quiet = false;
  // --perf-csv <path>: also write the measured metrics as CSV. Empty = off.
  std::string                              perf_csv;
  // When true, the Runner already streamed the per-test lines live (to stderr)
  // as each test finished, so the Reporter must NOT re-print the end-block (it
  // would double-print). The summary is always printed regardless. Set by the
  // single-node path; left false for multi-rank (so the reconciled cross-rank
  // end-block still prints).
  bool                                     per_test_already_streamed = false;
  // The ranks/hosts that participated, emitted as report.json `nodes[]`. Each
  // launcher populates this with the actual (rank, host) pairs it ran — one
  // localhost entry for the single-node path, two for --local-ranks 2, and one
  // per node for --nodes. When left empty the Reporter falls back to a single
  // localhost/rank-0 entry.
  // The per-node descriptor carried into report.json `nodes[].environment`. It is
  // populated by the launcher from each worker's `@env` block (or detected
  // locally for the single-node path). `in.vendor` stays the POLICY lookup key;
  // environment.vendor is the DETECTED vendor and may differ when --vendor
  // overrides the policy key.
  struct NodeInfo {
    int             rank = 0;
    std::string     host;
    EnvironmentInfo environment;
  };
  std::vector<NodeInfo>                    nodes;
  // Per-rank raw worker results: test id -> (rank -> TestResult). Reconciliation
  // (§7.3) reduces over the ranks that produced a row for a given test; a rank
  // absent from a test's map contributes nothing (treated as not-present). For a
  // single-node run this holds rank 0 only -> reduced_from {"0": status}; for a
  // 2-rank run it holds {0,1} (only the ranks that emitted a row).
  std::map<std::string, std::map<int, TestResult>> rank_results;

  // Convenience for the single-node path: record a rank-0 result for `id`.
  void add_result(const std::string& id, const TestResult& r) {
    rank_results[id][0] = r;
  }
};

// Select the EnvironmentInfo used for `when` predicate evaluation: the
// LOWEST-rank node's environment (rank 0 in normal runs), chosen explicitly so
// the result is independent of node insertion order (which is not guaranteed
// across launchers). Returns a reference into `nodes`; when `nodes` is empty it
// returns a reference to a static empty EnvironmentInfo (only catch-all `when`
// rows match). Exposed for testing the determinism guarantee.
//
// LIMITATION: the Reporter re-classifies tests using rank 0's environment,
// while each worker classified tests using its own local environment inside
// classify_and_partition(). On a heterogeneous cluster where rank 0 and rank N
// see different ROCm/fw versions, these two classifications can disagree — e.g.
// rank 1 ran a test as MUST_PASS (its env matched) while the Reporter would have
// gated it NA (rank 0's env didn't). The practical impact is low for homogeneous
// clusters (the common case); a future fix would reconcile per-rank
// classifications rather than re-applying one env to the reconciled results.
const EnvironmentInfo& pick_predicate_env(
    const std::vector<ReportInputs::NodeInfo>& nodes);

// Assemble the report.json document, write it to `report_path`, print a human
// summary, and return the gate exit code (0 pass / 1 fail). `policy` + `vendor`
// + the registry drive NA synthesis and classification.
int write_report(const Policy& policy, const ReportInputs& in,
                 const std::string& report_path);

}  // namespace acceptance
