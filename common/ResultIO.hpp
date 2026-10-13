/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// ResultIO — read/write the worker's per-rank results file (--results-out).
//
// Format (key=value lines — NOT JSON):
//   rank=<N>
//   @env        (optional EnvironmentInfo block, EnvironmentInfo::to_kv lines)
//   @test       (one block per result, each a TestResult::to_kv record)
//   ...
//
// Reusing the TestResult (de)serializer keeps the EXEC single-record file and
// this multi-record file in lockstep. The `@env` block is back-compatible: a
// stream without one parses to an empty EnvironmentInfo.

#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/TestResult.hpp"

namespace acceptance {

namespace result_io {

// Serialize {rank, results} to the per-rank file string (no @env block).
std::string serialize(int rank, const std::vector<TestResult>& results);

// Serialize {rank, env, results}: emits the `rank=` header, then an `@env` block
// carrying env.to_kv(), then the `@test` blocks. Back-compatible with the 2-arg
// form (a reader that ignores @env still recovers rank + results).
std::string serialize(int rank, const std::vector<TestResult>& results,
                      const EnvironmentInfo& env);

// Parse a per-rank file's text. On success fills out_rank + out_results and
// returns true. Returns false only if the `rank=` header is missing (a
// structural error the launcher treats as a bootstrap/run failure).
bool parse(const std::string& text, int& out_rank,
           std::vector<TestResult>& out_results);

// Parse overload that also surfaces the `@env` block into out_env. An absent
// `@env` block yields an empty EnvironmentInfo (back-compat).
bool parse(const std::string& text, int& out_rank,
           std::vector<TestResult>& out_results, EnvironmentInfo& out_env);

}  // namespace result_io

}  // namespace acceptance
