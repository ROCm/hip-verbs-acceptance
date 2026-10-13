/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_perf — the PerfSummary key grammar, rank merge, console rendering and CSV
// quoting. These are pure string functions on the boundary between what a test
// emits (Perf.hpp's flat keys) and what a human/CI reads, so they are worth
// pinning independently of any GPU or NIC being present.
//
// The key grammar is duplicated in scripts/perf_report.py; the cases below are
// the contract both sides must honour.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/PerfSummary.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

namespace accept_test {

namespace {

// Build the flat key group Perf.hpp emits for one measured region.
PerfSink emit_group(const std::string& metric, const std::string& size,
                    const char* unit, const char* median, const char* min,
                    const char* max, int iters) {
  const std::string suffix = size.empty() ? "" : "." + size;
  return {
      {metric + ".median" + suffix, median},
      {metric + ".min" + suffix, min},
      {metric + ".max" + suffix, max},
      {metric + ".avg" + suffix, median},
      {metric + ".unit" + suffix, unit},
      {metric + ".iters" + suffix, std::to_string(iters)},
      {metric + ".samples" + suffix + "." + unit, "[1.0, 2.0]"},
  };
}

// Join a rendered section into one blob so a case can assert on substrings.
std::string joined(const std::vector<std::string>& lines) {
  std::string out;
  for (const std::string& l : lines) out += l + "\n";
  return out;
}

void test_parse_shapes() {
  // Swept metric: size split out, unit read from the .unit key.
  PerfSink flat = emit_group("rdma_write.bw", "4KiB", "GB/s", "12.5", "12.0",
                             "13.0", 10);
  std::vector<PerfMetric> got = parse_perf_metrics(flat);
  CHECK(got.size() == 1, "swept metric collapses to one group");
  if (got.size() == 1) {
    CHECK_EQ_STR(got[0].metric, "rdma_write.bw", "metric name");
    CHECK_EQ_STR(got[0].size, "4KiB", "size tag");
    CHECK_EQ_STR(got[0].unit, "GB/s", "unit");
    CHECK(got[0].has_median && got[0].median == 12.5, "median value");
    CHECK(got[0].has_min && got[0].min == 12.0, "min value");
    CHECK(got[0].has_max && got[0].max == 13.0, "max value");
    CHECK(got[0].iters == 10, "iters");
    CHECK(got[0].size_bytes() == 4096, "size_bytes");
  }

  // Unswept metric: no size component, unit still recovered.
  std::vector<PerfMetric> plain = parse_perf_metrics(
      emit_group("qp_create_destroy.lat", "", "us/op", "8.25", "8.0", "9.0", 10));
  CHECK(plain.size() == 1, "unswept metric collapses to one group");
  if (plain.size() == 1) {
    CHECK_EQ_STR(plain[0].metric, "qp_create_destroy.lat", "unswept metric name");
    CHECK_EQ_STR(plain[0].size, "", "unswept has no size");
    CHECK_EQ_STR(plain[0].unit, "us/op", "unswept unit");
    CHECK(plain[0].size_bytes() == 0, "unswept size_bytes is 0");
  }

  // The unit can be recovered from the .samples key alone (that is where it
  // rides when the group is trimmed).
  std::vector<PerfMetric> from_samples = parse_perf_metrics(
      {{"reg_mr.lat.median.1MiB", "42.0"},
       {"reg_mr.lat.samples.1MiB.us/op", "[41.0, 43.0]"}});
  CHECK(from_samples.size() == 1, "samples-only group parses");
  if (from_samples.size() == 1)
    CHECK_EQ_STR(from_samples[0].unit, "us/op", "unit from samples key");
}

void test_parse_tolerance() {
  // A key with no stat token is skipped, not fatal. The PerfProbe per-HIP_CHECK
  // timings ("<expr>.us") land here by design: one per call site would bury the
  // measured metrics.
  std::vector<PerfMetric> got = parse_perf_metrics({
      {"HIP_CHECK(hipMalloc(&p, n)).us", "3.100"},
      {"", ""},
      {"median", "1.0"},                       // stat token with no metric
      {"good.bw.median.1MiB", "5.0"},
      {"good.bw.median.1MiB.extra", "nonsense"},  // unparseable value
  });
  CHECK(got.size() == 1, "only the well-formed metric survives");
  if (got.size() == 1) CHECK_EQ_STR(got[0].metric, "good.bw", "surviving metric");
}

void test_size_ordering() {
  // Sizes must order numerically, not lexically: 64B < 4KiB < 1MiB < 256MiB.
  PerfSink flat;
  for (const char* size : {"1MiB", "64B", "256MiB", "4KiB"}) {
    PerfSink one = emit_group("bw", size, "GB/s", "1.0", "1.0", "1.0", 10);
    flat.insert(flat.end(), one.begin(), one.end());
  }
  std::vector<PerfMetric> got = parse_perf_metrics(flat);
  CHECK(got.size() == 4, "four sizes, four groups");
  if (got.size() == 4) {
    CHECK_EQ_STR(got[0].size, "64B", "first size");
    CHECK_EQ_STR(got[1].size, "4KiB", "second size");
    CHECK_EQ_STR(got[2].size, "1MiB", "third size");
    CHECK_EQ_STR(got[3].size, "256MiB", "fourth size");
  }

  CHECK(parse_size_tag("256MiB") == 256ull * 1024 * 1024, "parse 256MiB");
  CHECK(parse_size_tag("1500B") == 1500, "parse 1500B");
  CHECK(parse_size_tag("GB/s") == 0, "a unit is not a size");
  CHECK(parse_size_tag("us/op") == 0, "us/op is not a size");
  CHECK(parse_size_tag("KiB") == 0, "a bare suffix is not a size");
  CHECK(parse_size_tag("0MiB") == 0, "zero-size tag rejected");
}

void test_rank_merge() {
  PerfSink r0 = {{"bw.median.1MiB", "10.0"}};
  PerfSink r1 = {{"bw.median.1MiB", "9.0"}};

  // One reporting rank round-trips untouched — single-node output must not
  // change shape just because the merge exists.
  PerfSink solo = merge_rank_perf({{0, r0}});
  CHECK(solo.size() == 1, "solo merge keeps one key");
  if (solo.size() == 1)
    CHECK_EQ_STR(solo[0].first, "bw.median.1MiB", "solo key unprefixed");

  // Two ranks: the lowest keeps the bare key, the other is namespaced.
  PerfSink both = merge_rank_perf({{0, r0}, {1, r1}});
  CHECK(both.size() == 2, "both ranks survive the merge");
  if (both.size() == 2) {
    CHECK_EQ_STR(both[0].first, "bw.median.1MiB", "rank 0 key unprefixed");
    CHECK_EQ_STR(both[1].first, "rank1.bw.median.1MiB", "rank 1 key prefixed");
    CHECK_EQ_STR(both[1].second, "9.0", "rank 1 value preserved");
  }

  // When rank 0 measured nothing, rank 1 becomes the primary: the bare keys
  // always belong to the lowest rank that actually reported.
  PerfSink r1_only = merge_rank_perf({{0, PerfSink{}}, {1, r1}});
  CHECK(r1_only.size() == 1, "empty rank contributes nothing");
  if (r1_only.size() == 1)
    CHECK_EQ_STR(r1_only[0].first, "bw.median.1MiB",
                 "lowest REPORTING rank is unprefixed");

  // A merged map still parses: the prefix reads as part of the metric name.
  std::vector<PerfMetric> parsed = parse_perf_metrics(both);
  CHECK(parsed.size() == 2, "merged map yields two metrics");
  if (parsed.size() == 2)
    CHECK_EQ_STR(parsed[1].metric, "rank1.bw", "prefixed metric name");
}

void test_render() {
  std::vector<PerfTestRow> rows = {
      {"verbs/datapath", "vat_rdma_write",
       parse_perf_metrics(emit_group("rdma_write.bw", "1MiB", "GB/s", "21.44",
                                     "20.0", "23.0", 10))},
  };
  const std::vector<std::string> lines = render_perf_section(rows);
  const std::string text = joined(lines);

  CHECK(text.find("--- performance ---") != std::string::npos, "section header");
  CHECK(lines.size() > 2 && lines[2].find("min") != std::string::npos &&
            lines[2].find("max") != std::string::npos,
        "column header names min and max");
  CHECK(text.find("--- verbs/datapath ---") != std::string::npos,
        "category header matches the functional dash style");
  CHECK(text.find("vat_rdma_write") != std::string::npos, "test line");
  CHECK(text.find("rdma_write.bw") != std::string::npos, "metric line");
  CHECK(text.find("21.44") != std::string::npos, "median rendered");
  CHECK(text.find("20") != std::string::npos, "min rendered");
  CHECK(text.find("23") != std::string::npos, "max rendered");

  // A single-pass metric still gets explicit min/max columns (equal to the
  // median) rather than a special-cased "single sample" string.
  std::vector<PerfTestRow> single = {
      {"hip/memory", "Vmm_Create_Map_Roundtrip",
       parse_perf_metrics(
           emit_group("vmm.map_unmap", "", "us/op", "5.0", "5.0", "5.0", 1))}};
  CHECK(joined(render_perf_section(single)).find("vmm.map_unmap") !=
            std::string::npos,
        "single-sample metric still renders");

  // Two tests in the same category get a blank line between their metric
  // blocks — the newline-per-test-group behaviour.
  std::vector<PerfTestRow> two = {
      {"hip/memory", "Test1",
       parse_perf_metrics(emit_group("a.bw", "", "GB/s", "1.0", "1.0", "1.0", 5))},
      {"hip/memory", "Test2",
       parse_perf_metrics(emit_group("b.bw", "", "GB/s", "2.0", "2.0", "2.0", 5))},
  };
  CHECK(joined(render_perf_section(two)).find("\n\n  Test2") != std::string::npos,
        "blank line separates test groups within a category");

  // A rank-prefixed metric name ("rank1.foo") is longer than the base column
  // width; it must not push just its own row out of alignment with the rest.
  std::vector<PerfMetric> plain_metric = parse_perf_metrics(
      emit_group("vmm.posixfd.export_import", "", "us/op", "162.5", "153.6",
                "167.8", 10));
  std::vector<PerfMetric> ranked_metric = parse_perf_metrics(
      emit_group("rank1.vmm.posixfd.export_import", "", "us/op", "159.0",
                "155.5", "163.5", 10));
  std::vector<PerfTestRow> ranked = {{"hip/memmgr",
                                      "Vmm_PosixFd_Export_Import",
                                      {plain_metric[0], ranked_metric[0]}}};
  std::size_t first_pos = std::string::npos, second_pos = std::string::npos;
  int seen = 0;
  for (const std::string& line : render_perf_section(ranked)) {
    std::size_t pos = line.find("us/op");
    if (pos == std::string::npos) continue;
    if (seen == 0) first_pos = pos;
    if (seen == 1) second_pos = pos;
    ++seen;
  }
  CHECK(seen == 2, "both metric rows rendered");
  CHECK(first_pos != std::string::npos && first_pos == second_pos,
        "rank-prefixed metric name doesn't break column alignment");

  // Nothing measured -> no header at all, so a functional run stays clean.
  CHECK(render_perf_section({}).empty(), "empty input renders nothing");
  CHECK(render_perf_section({{"cat", "id", {}}}).empty(),
        "rows without metrics render nothing");
}

void test_csv() {
  // A metric name can carry a comma (a test is free to name one); the CSV must
  // quote it rather than shift every following column.
  std::vector<PerfTestRow> rows = {
      {"hip/data,movement", "Memcpy_Size_Sweep",
       parse_perf_metrics(
           emit_group("h2d.sync.bw", "64KiB", "GB/s", "12.5", "12.0", "13.0", 10))}};

  const std::string path = "/tmp/unit_perf_csv_check.csv";
  CHECK(write_perf_csv(path, rows), "csv writes");

  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  CHECK(text.rfind("category,test,metric,size,size_bytes,", 0) == 0,
        "csv starts with its header");
  CHECK(text.find("\"hip/data,movement\"") != std::string::npos,
        "comma-bearing field is quoted");
  CHECK(text.find(",65536,") != std::string::npos, "size_bytes column");
  CHECK(text.find(",GB/s,10") != std::string::npos, "unit and iters columns");
  std::remove(path.c_str());

  // An empty run still produces a well-formed, header-only file.
  const std::string empty_path = "/tmp/unit_perf_csv_empty.csv";
  CHECK(write_perf_csv(empty_path, {}), "empty csv writes");
  std::ifstream empty_in(empty_path);
  std::string header;
  std::getline(empty_in, header);
  CHECK(header.rfind("category,test,metric", 0) == 0, "header-only file");
  CHECK(!std::getline(empty_in, header), "no rows after the header");
  std::remove(empty_path.c_str());
}

}  // namespace

void run_perf_tests() {
  test_parse_shapes();
  test_parse_tolerance();
  test_size_ordering();
  test_rank_merge();
  test_render();
  test_csv();
}

}  // namespace accept_test
