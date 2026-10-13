/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Policy — the classification engine (ARCHITECTURE.md §7). Loads a per-vendor
// CSV policy file and resolves the Class of a (test_id, env) pair.
//
// The policy file is the authoritative run list. Unlisted tests are
// not run. Discovery mode (no policy file) runs all tests as MUST_PASS with
// the gate active.
//
// CSV format (split on the first 5 commas):
//   test,class,when,expected,ticket,reason
// `when` is an optional predicate (empty = always applies; see eval_when in
// Policy.cpp). `reason` is the verbatim remainder (may contain commas). `#`
// lines and blank lines are skipped. `class` is parsed case-insensitively.
// Within a file, for each matching test id / glob, rows are tried in
// declaration order and the first row whose `when` predicate evaluates true
// wins; this allows the piecewise pattern:
//   test,SKIP,rocm<7.0,,TICKET,old stack
//   test,MUST_PASS,,,,
// All policy files must use the 6-field format above. A row with fewer than 5
// commas is parsed positionally with missing trailing fields defaulting to empty;
// a row with more than 5 commas has the extra commas absorbed into `reason`.
//
// SKIP inference: when a test is in-policy but its `when` predicate
// fails and that predicate is software-gated (rocm/driver/fw/sym dims only),
// the result is SKIP rather than NA. Use explicit NA rows for hardware-permanent
// conditions (no `when`, or a hardware-dim-only predicate).

#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/Types.hpp"

namespace acceptance {

// Resolved classification for one test.
struct Classification {
  Class       cls = Class::Na;
  std::string reason;
  std::string ticket;
  std::string expected_status;  // KNOWN_FAILURE only: expected reduced status,
                                // one of "fail"|"pass"|"skip"|"none"; empty => "fail".
  std::string when_matched;     // firing predicate, or empty for catch-all rows.
};

struct Policy {
  bool present = false;  // false => discovery mode (no policy file in effect)

  // Load a policy directly from `path` (a .csv file). `explicit_flag` is true
  // when the user passed --policy explicitly:
  //   * !explicit_flag && file missing -> Policy{present=false} (discovery/WARN)
  //   *  explicit_flag && file missing -> throw std::runtime_error
  static Policy load(const std::string& path, bool explicit_flag);

  // Resolve the class for (test_id, env). Resolution order:
  //   1. Exact-id rows (declaration order): first whose `when` passes wins.
  //   2. Else glob rows (longest literal prefix; ties -> declaration order):
  //      first whose `when` passes wins.
  //   3. In-policy but no predicate passed:
  //      - `when` is a software-dim predicate (rocm/driver/fw/sym) -> SKIP
  //      - otherwise -> NA
  //   4. Truly unlisted -> `found` set to false; caller excludes test from run.
  // NOTE: an exact id match (even with a failing predicate) shadows all globs.
  Classification classify(const std::string& test_id,
                          const EnvironmentInfo& env,
                          bool* found = nullptr) const;

 private:
  struct Entry {
    std::string    test;  // exact id or glob pattern
    std::string    when;  // when predicate (empty = always applies)
    Classification cls;
  };

  static Entry parse_row(const std::string& row);

  std::vector<Entry> entries_;  // only meaningful when present == true
};

}  // namespace acceptance
