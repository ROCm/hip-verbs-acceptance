/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// PerfSummary — read the flat perf key=value pairs a test hands back and turn
// them into something a human (console block) or a spreadsheet (CSV) can use.
//
// Perf.hpp writes metrics as flat strings so TestResult stays a line-oriented
// key=value record (ADR-002). One measured region emits a group of keys:
//
//   <metric>.median[.<size>]           <metric>.unit[.<size>]
//   <metric>.min[.<size>]              <metric>.iters[.<size>]
//   <metric>.max[.<size>]              <metric>.samples[.<size>].<unit>
//   <metric>.avg[.<size>]
//
// This header reassembles those groups. It is the C++ twin of
// scripts/perf_report.py's split_key()/parse_perf(); the two must agree on the
// key grammar, so change them together.
//
// Nothing here is HIP- or verbs-specific: it operates on strings only, which is
// why it can link into the unit_tests target.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "common/Perf.hpp"

namespace acceptance {

// One (metric, size) measurement group, reassembled from its flat keys. A stat
// that never appeared in the input keeps its zero and is reported absent via
// the has_* flag, so a missing median is distinguishable from a measured 0.
struct PerfMetric {
  std::string metric;      // "rdma_write.bw"
  std::string size;        // "4KiB", empty when the metric is not swept
  std::string unit;        // "GB/s", "us/op", empty when unknown
  double      median = 0;
  double      min = 0;
  double      max = 0;
  double      avg = 0;
  long long   iters = 0;
  bool        has_median = false;
  bool        has_min = false;
  bool        has_max = false;
  bool        has_avg = false;

  // The size in bytes, for numeric ordering (64B < 4KiB < 1MiB). 0 when the
  // metric carries no size.
  std::size_t size_bytes() const;
};

// "256MiB" -> 268435456, "1500B" -> 1500; 0 when `tag` is not a size tag (the
// inverse of Perf.hpp's perf_detail::human_bytes). Exposed because the key
// splitter uses it to tell a size component from a unit component.
std::size_t parse_size_tag(const std::string& tag);

// All metrics reported by one test, in display order.
struct PerfTestRow {
  std::string             category;  // "hip/datamovement"
  std::string             id;        // "Memcpy_Size_Sweep"
  std::vector<PerfMetric> metrics;
};

// Reassemble `flat` into metric groups, sorted by (metric, size_bytes).
// Unrecognised keys are skipped rather than fatal: the perf map is produced by
// a worker process that may be a different build, and a surprise in it must not
// take down the report.
std::vector<PerfMetric> parse_perf_metrics(const PerfSink& flat);

// Merge the per-rank perf maps of ONE test into a single flat map. The lowest
// rank that reported keeps its keys unchanged; every other rank's keys are
// prefixed "rank<N>." so the two do not collide. `per_rank` is ordered by rank
// (a std::map iteration in the caller), and ranks that reported nothing are
// skipped. A single reporting rank therefore round-trips unchanged.
PerfSink merge_rank_perf(const std::vector<std::pair<int, PerfSink>>& per_rank);

// Render the end-of-run console block, one line per metric, grouped by category
// and test. Returns an empty vector when `rows` holds no metrics at all, so the
// caller can skip the header.
std::vector<std::string> render_perf_section(const std::vector<PerfTestRow>& rows);

// Write `rows` as CSV (header + one row per metric) to `path`. Returns false and
// leaves an error on stderr when the file cannot be written. Always writes the
// header, so an empty run still produces a well-formed file.
bool write_perf_csv(const std::string& path, const std::vector<PerfTestRow>& rows);

}  // namespace acceptance
