/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/PerfSummary.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <map>
#include <system_error>
#include <tuple>

namespace acceptance {

namespace {

// Stat tokens emitted by perf_detail::emit (Perf.hpp). A metric name never
// contains one, so the first match scanning a dotted key left-to-right splits
// "<metric>" from "<stat>[.<size>][.<unit>]".
const char* const kStats[] = {"median", "min",   "max",    "avg",
                              "unit",   "iters", "samples"};

bool is_stat(const std::string& part) {
  for (const char* s : kStats)
    if (part == s) return true;
  return false;
}

// The binary units human_bytes() produces, longest-first so "KiB" is tried
// before "B" (a suffix of nothing here, but the order documents the intent).
struct Unit { const char* suffix; std::size_t scale; };
const Unit kUnits[] = {{"KiB", 1024ull},
                       {"MiB", 1024ull * 1024},
                       {"GiB", 1024ull * 1024 * 1024},
                       {"TiB", 1024ull * 1024 * 1024 * 1024},
                       {"B", 1}};

// Split a dotted key. Returns false when it carries no stat token — which is the
// case for the PerfProbe per-HIP_CHECK timings ("<expr>.us"). Those are
// deliberately left out of the console block and the CSV: perf mode records one
// per HIP_CHECK call site, so surfacing them would bury the measured metrics
// under hundreds of lines. They remain in report.json for anyone who wants them
// (scripts/perf_report.py skips them for the same reason).
bool split_key(const std::string& key, std::string* metric, std::string* stat,
               std::string* size, std::string* unit) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (true) {
    const std::size_t dot = key.find('.', start);
    parts.push_back(key.substr(start, dot - start));
    if (dot == std::string::npos) break;
    start = dot + 1;
  }

  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (!is_stat(parts[i])) continue;

    metric->clear();
    for (std::size_t j = 0; j < i; ++j) {
      if (j) *metric += ".";
      *metric += parts[j];
    }
    if (metric->empty()) return false;  // a key that is only a stat token

    *stat = parts[i];
    std::size_t rest = i + 1;
    size->clear();
    if (rest < parts.size() && parse_size_tag(parts[rest]) > 0) {
      *size = parts[rest];
      ++rest;
    }
    // Whatever trails is the unit (units contain '/' but never '.').
    unit->clear();
    for (std::size_t j = rest; j < parts.size(); ++j) {
      if (j > rest) *unit += ".";
      *unit += parts[j];
    }
    return true;
  }
  return false;
}

// "12.340" -> 12.34; false when the string is not a bare number end-to-end.
bool parse_double(const std::string& s, double* out) {
  const char* first = s.data();
  const char* last = s.data() + s.size();
  auto res = std::from_chars(first, last, *out);
  return res.ec == std::errc{} && res.ptr == last;
}

bool parse_ll(const std::string& s, long long* out) {
  const char* first = s.data();
  const char* last = s.data() + s.size();
  auto res = std::from_chars(first, last, *out);
  return res.ec == std::errc{} && res.ptr == last;
}

// Shortest round-trip decimal, locale-independent (CSV wants the full value,
// not the console's rounded form).
std::string exact(double v) {
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  if (res.ec != std::errc{}) return "0";
  return std::string(buf, res.ptr);
}

// Console number: 3 decimals with the noise digits trimmed below 1000, comma-
// grouped integers above it. Mirrors perf_report.py's num().
std::string compact(double v) {
  if (v >= 1000.0 || v <= -1000.0) {
    std::string s = perf_detail::format_fixed(v, 0);
    const std::size_t head = (s[0] == '-') ? 1u : 0u;
    for (std::size_t pos = s.size(); pos > head + 3;) {
      pos -= 3;
      s.insert(pos, ",");
    }
    return s;
  }
  std::string s = perf_detail::format_fixed(v, 3);
  while (!s.empty() && s.back() == '0') s.pop_back();
  if (!s.empty() && s.back() == '.') s.pop_back();
  return s.empty() ? "0" : s;
}

// Left-pad to `width` (no truncation: a long value pushes the row wider rather
// than losing digits).
std::string rpad(const std::string& s, std::size_t width) {
  return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

std::string lpad(const std::string& s, std::size_t width) {
  return s.size() >= width ? s : std::string(width - s.size(), ' ') + s;
}

// A CSV field: quoted only when it must be (comma, quote, or newline inside).
std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
  std::string out = "\"";
  for (char c : s) {
    out += c;
    if (c == '"') out += '"';
  }
  return out + "\"";
}

}  // namespace

// "256MiB" -> 268435456, "1500B" -> 1500, 0 when `tag` is not a size.
std::size_t parse_size_tag(const std::string& tag) {
  for (const Unit& u : kUnits) {
    const std::size_t len = std::string(u.suffix).size();
    if (tag.size() <= len) continue;
    if (tag.compare(tag.size() - len, len, u.suffix) != 0) continue;

    const std::string digits = tag.substr(0, tag.size() - len);
    for (char c : digits)
      if (c < '0' || c > '9') return 0;

    long long value = 0;
    if (!parse_ll(digits, &value) || value <= 0) return 0;
    return static_cast<std::size_t>(value) * u.scale;
  }
  return 0;
}

std::size_t PerfMetric::size_bytes() const { return parse_size_tag(size); }

std::vector<PerfMetric> parse_perf_metrics(const PerfSink& flat) {
  // Keyed by (metric, size bytes) so the map iteration is already the display
  // order: alphabetical by metric, then numeric by size (64B before 4KiB).
  std::map<std::pair<std::string, std::size_t>, PerfMetric> groups;

  for (const auto& [key, value] : flat) {
    std::string metric, stat, size, unit;
    if (!split_key(key, &metric, &stat, &size, &unit)) continue;

    PerfMetric& m = groups[{metric, parse_size_tag(size)}];
    m.metric = metric;
    m.size = size;

    if (stat == "unit") {
      m.unit = value;
    } else if (stat == "samples") {
      // The unit rides in the key for the raw array; the array itself is not
      // reduced here (median/min/max already carry the summary).
      if (m.unit.empty()) m.unit = unit;
    } else if (stat == "iters") {
      parse_ll(value, &m.iters);
    } else {
      double d = 0.0;
      if (!parse_double(value, &d)) continue;
      if (stat == "median") { m.median = d; m.has_median = true; }
      else if (stat == "min") { m.min = d; m.has_min = true; }
      else if (stat == "max") { m.max = d; m.has_max = true; }
      else if (stat == "avg") { m.avg = d; m.has_avg = true; }
    }
  }

  std::vector<PerfMetric> out;
  out.reserve(groups.size());
  for (auto& [_, m] : groups) out.push_back(std::move(m));
  return out;
}

PerfSink merge_rank_perf(const std::vector<std::pair<int, PerfSink>>& per_rank) {
  PerfSink merged;
  bool primary_taken = false;

  for (const auto& [rank, sink] : per_rank) {
    if (sink.empty()) continue;
    if (!primary_taken) {
      // Lowest reporting rank owns the bare keys, so single-rank runs (and the
      // rank-0-measures case) look exactly as they did before.
      primary_taken = true;
      merged.insert(merged.end(), sink.begin(), sink.end());
      continue;
    }
    const std::string prefix = "rank" + std::to_string(rank) + ".";
    for (const auto& [key, value] : sink)
      merged.emplace_back(prefix + key, value);
  }
  return merged;
}

// Column header for the metric table, kept in the same padding as the data
// rows below it so the two stay aligned.
std::string perf_header_line(std::size_t metric_width) {
  return "    " + rpad("metric", metric_width) + lpad("size", 8) + "  " +
         lpad("median", 10) + " " + lpad("min", 10) + " " + lpad("max", 10) +
         " " + rpad("unit", 6) + " " + lpad("n", 4);
}

std::vector<std::string> render_perf_section(
    const std::vector<PerfTestRow>& rows) {
  std::vector<std::string> out;
  std::size_t total = 0;
  // A rank-prefixed metric ("rank1.foo.bar") can be longer than the base
  // width; size the column to the widest name in this render, or every row
  // but that one silently drifts out of alignment.
  std::size_t metric_width = 26;
  for (const PerfTestRow& row : rows) {
    total += row.metrics.size();
    for (const PerfMetric& m : row.metrics)
      metric_width = std::max(metric_width, m.metric.size());
  }
  if (total == 0) return out;

  out.push_back("");
  out.push_back("--- performance ---");
  out.push_back(perf_header_line(metric_width));

  std::string last_category;
  bool first_test = true;
  for (const PerfTestRow& row : rows) {
    if (row.metrics.empty()) continue;
    if (row.category != last_category) {
      last_category = row.category;
      out.push_back("");
      out.push_back("--- " + row.category + " ---");
    } else if (!first_test) {
      out.push_back("");  // separates this test's group from the previous one
    }
    first_test = false;
    out.push_back("  " + row.id);

    for (const PerfMetric& m : row.metrics) {
      out.push_back(
          "    " + rpad(m.metric, metric_width) + lpad(m.size, 8) + "  " +
          lpad(m.has_median ? compact(m.median) : "-", 10) + " " +
          lpad(m.has_min ? compact(m.min) : "-", 10) + " " +
          lpad(m.has_max ? compact(m.max) : "-", 10) + " " + rpad(m.unit, 6) +
          " " + lpad(m.iters > 0 ? std::to_string(m.iters) : "-", 4));
    }
  }
  out.push_back("");
  return out;
}

bool write_perf_csv(const std::string& path,
                    const std::vector<PerfTestRow>& rows) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "error: cannot open perf csv '%s'\n", path.c_str());
    return false;
  }

  out << "category,test,metric,size,size_bytes,median,min,max,avg,unit,iters\n";
  for (const PerfTestRow& row : rows) {
    for (const PerfMetric& m : row.metrics) {
      out << csv_field(row.category) << "," << csv_field(row.id) << ","
          << csv_field(m.metric) << "," << csv_field(m.size) << ","
          << m.size_bytes() << "," << (m.has_median ? exact(m.median) : "")
          << "," << (m.has_min ? exact(m.min) : "") << ","
          << (m.has_max ? exact(m.max) : "") << ","
          << (m.has_avg ? exact(m.avg) : "") << ","
          << csv_field(m.unit) << "," << m.iters << "\n";
    }
  }

  out.flush();
  if (!out) {
    std::fprintf(stderr, "error: failed writing perf csv '%s'\n", path.c_str());
    return false;
  }
  return true;
}

}  // namespace acceptance
