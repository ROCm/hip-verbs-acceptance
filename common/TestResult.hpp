/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// TestResult — the single per-rank outcome an EXEC child serializes to its
// --result-file. The runner (Step 2) reads it back and may overwrite/synthesize
// fields (exit_code, synth) when the child misbehaves.
//
// Serialization: a flat key=value line format — NOT JSON. We never
// parse JSON anywhere in the binary; the only JSON we produce is the external
// report.json (hand-emitted by the Reporter). One TestResult serializes as:
//
//   id=<id>
//   status=<pass|fail|skip|none>
//   reason=<free-form, may contain '=' and ','>
//   duration_ms=<double>
//   exit_code=<int>
//   synth=<crash|timeout|corrupt|"">
//
// Parse rule: each line is split on the FIRST '='; the key is the text before it
// and the value is everything after (so reasons may contain '='). Values are
// single-line (empty allowed). Missing keys take sensible defaults; unknown keys
// are ignored. This deserializer never throws — a malformed/empty file simply
// yields defaults; the Runner detects the problem via the id mismatch (§6.3).

#include <charconv>
#include <istream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "common/Types.hpp"

namespace acceptance {

struct TestResult {
  std::string id{};
  Status      status = Status::Pass;
  std::string reason{};          // populated for fail/skip
  double      duration_ms = 0;
  int         exit_code = 0;   // set by runner for synthesized rows
  std::string synth{};           // "", "crash", "timeout", "corrupt" (runner-set)

  // Perf metrics as (key, value) pairs, serialized as perf.<key>=<value> lines.
  // Filled by the perf helpers under --kind perf|all; empty otherwise.
  std::vector<std::pair<std::string, std::string>> perf{};

  // Replace newlines/carriage-returns with spaces so a value can never break the
  // line-oriented key=value format (a multi-line reason would otherwise inject a
  // phantom record / truncate the field). Values are single-line by contract;
  // this enforces it defensively.
  static std::string one_line(std::string s) {
    for (char& c : s)
      if (c == '\n' || c == '\r') c = ' ';
    return s;
  }

  // Emit the key=value block (no trailing record separator). Doubles/ints are
  // written with std::to_chars so the output is LOCALE-INDEPENDENT (a comma-
  // decimal locale must not turn 0.4 into "0,4"). The companion report.json
  // emitter relies on the same guarantee.
  void to_kv(std::ostream& os) const {
    os << "id=" << one_line(id) << "\n";
    os << "status=" << to_string(status) << "\n";
    os << "reason=" << one_line(reason) << "\n";
    os << "duration_ms=" << double_to_chars(duration_ms) << "\n";
    os << "exit_code=" << int_to_chars(exit_code) << "\n";
    os << "synth=" << one_line(synth) << "\n";
    for (const auto& kv : perf)
      os << "perf." << one_line(kv.first) << "=" << one_line(kv.second) << "\n";
  }

  // Parse a key=value block from a list of lines. Tolerant: missing keys default,
  // unknown keys ignored, never throws.
  static TestResult from_kv(const std::vector<std::string>& lines) {
    TestResult r;
    for (const std::string& raw : lines) {
      // Strip a trailing '\r' (CRLF tolerance) but keep interior content.
      std::string line = raw;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      const std::size_t eq = line.find('=');
      if (eq == std::string::npos) continue;  // not a key=value line
      const std::string key = line.substr(0, eq);
      const std::string val = line.substr(eq + 1);
      if (key == "id")               r.id = val;
      else if (key == "status")      r.status = status_from(val);
      else if (key == "reason")      r.reason = val;
      else if (key == "duration_ms") r.duration_ms = parse_double(val);
      else if (key == "exit_code")   r.exit_code = parse_int(val);
      else if (key == "synth")       r.synth = val;
      else if (key.rfind("perf.", 0) == 0)
        r.perf.emplace_back(key.substr(5), val);
      // unknown keys ignored
    }
    return r;
  }

  // Parse a key=value block from a stream (reads to EOF). Used for the EXEC
  // child's single-record --result-file.
  static TestResult from_kv(std::istream& is) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(is, line)) lines.push_back(line);
    return from_kv(lines);
  }

  // --- locale-independent numeric formatting helpers ------------------------

  static std::string double_to_chars(double v) {
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), v);
    if (res.ec != std::errc{}) return "0";
    return std::string(buf, res.ptr);
  }

  static std::string int_to_chars(long long v) {
    char buf[32];
    auto res = std::to_chars(buf, buf + sizeof(buf), v);
    if (res.ec != std::errc{}) return "0";
    return std::string(buf, res.ptr);
  }

 private:
  static double parse_double(const std::string& s) {
    double out = 0.0;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto res = std::from_chars(first, last, out);
    if (res.ec != std::errc{}) return 0.0;
    return out;
  }

  static int parse_int(const std::string& s) {
    long long out = 0;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto res = std::from_chars(first, last, out);
    if (res.ec != std::errc{}) return 0;
    return static_cast<int>(out);
  }
};

}  // namespace acceptance
