/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/TestResult.hpp"

namespace acceptance {

using PerfSink = std::vector<std::pair<std::string, std::string>>;
using PerfClock = std::chrono::steady_clock;

inline bool& perf_mode_ref() { static bool on = false; return on; }
inline void set_perf_mode(bool on) { perf_mode_ref() = on; }
inline bool perf_probe_mode() { return perf_mode_ref(); }

struct MeasureOpts { int warmup = 3; int iters = 10; };

namespace perf_detail {

// Locale-independent fixed-decimal (a comma-locale must not turn 0.4 into "0,4").
inline std::string format_fixed(double value, int decimals) {
  if (decimals <= 0) {
    return std::to_string(static_cast<long long>(std::llround(value)));
  }
  long long factor = 1;
  for (int i = 0; i < decimals; ++i) {
    factor *= 10;
  }

  long long scaled = static_cast<long long>(std::llround(value * factor));
  const bool negative = scaled < 0;
  if (negative) {
    scaled = -scaled;
  }
  std::string frac = std::to_string(scaled % factor);
  frac.insert(frac.begin(), decimals - static_cast<int>(frac.size()), '0');
  return (negative ? "-" : "") + std::to_string(scaled / factor) + "." + frac;
}

// Linearly-interpolated quantile (fraction 0..1). Sorts `samples` in place.
inline double quantile(std::vector<double>& samples, double fraction) {
  if (samples.empty()) {
    return 0.0;
  }

  std::sort(samples.begin(), samples.end());
  const double pos = fraction * (static_cast<double>(samples.size()) - 1.0);
  const std::size_t lower = static_cast<std::size_t>(pos);
  const std::size_t upper = std::min(lower + 1, samples.size() - 1);

  return samples[lower] +
         (samples[upper] - samples[lower]) * (pos - static_cast<double>(lower));
}

// 268435456 -> "256MiB", 65536 -> "64KiB", 1500 -> "1500B" (readable key suffix).
inline std::string human_bytes(std::size_t bytes) {
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int tier = 0;
  while (tier < 4 && bytes >= 1024 && bytes % 1024 == 0) { bytes /= 1024; ++tier; }
  return std::to_string(bytes) + kUnits[tier];
}

// Reserved dotted-key components PerfSummary.cpp's split_key() (kStats) scans
// for to split "<metric>" from "<stat>[.<size>][.<unit>]". A metric name must
// never contain one of these as a component, or it silently mis-splits on the
// read side. Checked unconditionally (not assert()) because this codebase's
// perf builds run Release/NDEBUG, where assert() is compiled out — a metric
// name is a static, hardcoded programming error, not a data-dependent
// condition, so it should fail loud in every build. Keep in sync with
// PerfSummary.cpp's kStats.
inline void check_metric_name(const std::string& metric) {
  static const char* const kReservedStats[] = {"median", "min",   "max",    "avg",
                                                "unit",   "iters", "samples"};
  std::size_t start = 0;
  for (;;) {
    const std::size_t dot = metric.find('.', start);
    const std::string part = metric.substr(start, dot - start);
    for (const char* s : kReservedStats) {
      if (part == s) {
        std::fprintf(stderr,
                     "fatal: perf metric name \"%s\" has a component that "
                     "collides with a reserved stat token \"%s\" "
                     "(median/min/max/avg/unit/iters/samples) -- rename the "
                     "metric\n",
                     metric.c_str(), s);
        std::abort();
      }
    }
    if (dot == std::string::npos) break;
    start = dot + 1;
  }
}

// Append <metric>.<stat>[.<size>] rows (+ .unit/.iters/.samples) for `samples`
// to `sink`, leading with the median. `size` (0 == none) is humanized.
inline void emit(PerfSink& sink, const std::string& metric, std::size_t size,
                 std::vector<double> samples, const char* unit) {
  check_metric_name(metric);
  const std::string suffix = size ? "." + human_bytes(size) : std::string();
  const int decimals = (unit && std::strcmp(unit, "GB/s") == 0) ? 2 : 3;
  auto key = [&](const char* part) { return metric + "." + part + suffix; };
  auto num = [&](double value) { return format_fixed(value, decimals); };

  double total = 0.0;
  for (double value : samples) {
     total += value;
  }

  const double average =
      samples.empty() ? 0.0 : total / static_cast<double>(samples.size());

  // Raw per-run values, in run order, as a JSON array (before quantile sorts).
  std::string series = "[";
  for (std::size_t index = 0; index < samples.size(); ++index) {  
    series += (index ? ", " : "") + num(samples[index]);
  }
  series += "]";

  // Braced list is left-to-right: median sorts first, then min/max read the ends.
  const std::pair<const char*, double> stats[] = {
      {"median", quantile(samples, 0.50)},
      {"min", samples.empty() ? 0.0 : samples.front()},
      {"max", samples.empty() ? 0.0 : samples.back()},
      {"avg", average},
  };
  for (const auto& [stat, value] : stats) {
    sink.emplace_back(key(stat), num(value));
  }
  
  sink.emplace_back(key("unit"), unit);
  sink.emplace_back(key("iters"), std::to_string(samples.size()));
  // Unit folded into the key so the raw array is self-describing on its own.
  sink.emplace_back(key("samples") + "." + unit, series);
}

}  // namespace perf_detail

// HIP_CHECK auto-timing: an active PerfProbeScope records each call in us.
// Repeated calls get #N suffixes; keys are sanitized for key=value output.
struct PerfProbe {
  bool on = false;
  PerfSink* sink = nullptr;
  std::map<std::string, int> seen;
  bool active() const { return on && sink != nullptr; }
};

inline PerfProbe& perf_probe() { 
  static thread_local PerfProbe probe;
  return probe;
}

inline void perf_probe_record(const char* expr, double microseconds) {
  PerfProbe& probe = perf_probe();
  if (!probe.active()) {
    return;
  }

  std::string key = expr;
  for (char& c : key) {
    if (c == '=' || c == '\n' || c == '\r') {
      c = '_';
    }
  }

  if (++probe.seen[key] > 1) {
    key += "#" + std::to_string(probe.seen[key]);
  }
  probe.sink->emplace_back(key + ".us", perf_detail::format_fixed(microseconds, 3));
}

class PerfProbeScope {
 public:
  explicit PerfProbeScope(PerfSink& sink) : prev_(perf_probe()) {
    if (perf_probe_mode()) perf_probe() = PerfProbe{true, &sink, {}};
  }
  ~PerfProbeScope() { perf_probe() = std::move(prev_); }
  PerfProbeScope(const PerfProbeScope&) = delete;
  PerfProbeScope& operator=(const PerfProbeScope&) = delete;

 private:
  PerfProbe prev_;
};

// Reads the clock only when a probe is active, so a disabled probe is ~free.
class PerfTimer {
 public:
  explicit PerfTimer(const char* expr) {
    if (perf_probe().active()) {
      expr_ = expr;
      start_ = PerfClock::now();
    }
  }
  void stop() {
    if (expr_ != nullptr) {
      perf_probe_record(expr_, std::chrono::duration<double, std::micro>(
                                   PerfClock::now() - start_).count());
    }
    expr_ = nullptr;
  }

 private:
  const char* expr_ = nullptr;
  PerfClock::time_point start_;
};

// Unified functional/perf loop: one pass normally, warmup+N in perf mode.
// Each completed pass is timed; failed or interrupted passes are ignored.
class PerfLoop {
 public:
  explicit PerfLoop(MeasureOpts options = {}) : options_(options) {}

  int run_count() const {
    return perf_probe_mode() ? options_.warmup + options_.iters : 1;
  }

  // Current range-for pass. Its non-trivial destructor avoids unused-variable
  // warnings even when the loop body does not reference `pass`.
  struct Pass {
    int index = 0;
    ~Pass() {}
    bool first() const { return index == 0; }  // the sole functional pass
  };

  // Range-for iterator: starting a pass reads the clock, advancing records the
  // elapsed time (past warmup) and starts the next pass.
  class Iterator {
   public:
    Iterator(PerfLoop* owner, int index) : owner_(owner), index_(index) {
      if (index_ < owner_->run_count()) owner_->start_pass();
    }
    Pass operator*() const { return Pass{index_}; }
    bool operator!=(const Iterator& other) const {
      return index_ != other.index_;
    }
    Iterator& operator++() {
      owner_->finish_pass();
      if (++index_ < owner_->run_count()) {
        owner_->start_pass();
      }
      return *this;
    }

   private:
    PerfLoop* owner_;
    int index_;
  };

  Iterator begin() { return Iterator(this, 0); }
  Iterator end() { return Iterator(this, run_count()); }

  // GB/s over `bytes` (nothing with perf off). Unmeasurable passes (elapsed at/
  // below clock resolution) are dropped, not clamped to 0.0 that would skew .min.
  void emit_bandwidth(PerfSink& sink, const std::string& metric,
                      std::size_t bytes) {
    std::vector<double> rates;

    rates.reserve(samples_.size());
    for (double elapsed : samples_) {
      if (elapsed > 0.0) {
        rates.push_back(static_cast<double>(bytes) / elapsed / 1e9);
      }
    }

    samples_ = std::move(rates);
    flush(sink, metric, bytes, "GB/s");
  }

  // us/op; emits nothing with perf off.
  void emit_latency(PerfSink& sink, const std::string& metric,
                    std::size_t size = 0) {
    for (double& sample : samples_) {
      sample *= 1e6;
    }
    flush(sink, metric, size, "us/op");
  }

 private:
  void start_pass() { started_ = PerfClock::now(); }
  void finish_pass() {
    if (perf_probe_mode() && recorded_index_ >= options_.warmup) {
      samples_.push_back(
          std::chrono::duration<double>(PerfClock::now() - started_).count());
    }
    ++recorded_index_;
  }
  void flush(PerfSink& sink, const std::string& metric, std::size_t size,
             const char* unit) {
    if (!samples_.empty()) {
      perf_detail::emit(sink, metric, size, std::move(samples_), unit);
    }
  }

  MeasureOpts options_;
  std::vector<double> samples_;
  int recorded_index_ = 0;
  PerfClock::time_point started_;
};

// Return a passing TestResult carrying `sink` as its perf metrics. Consolidates
// the pass()+perf-move tail shared by tests that emit region metrics into a sink.
inline TestResult pass_with_perf(PerfSink sink) {
  TestResult result;
  result.status = Status::Pass;
  result.perf = std::move(sink);
  return result;
}

// Emit a PerfLoop region's latency and return a passing TestResult carrying it.
// The common tail for tests whose functional region is wrapped in a PerfLoop.
inline TestResult pass_with_latency(PerfLoop& loop, const std::string& metric) {
  PerfSink sink;
  loop.emit_latency(sink, metric);
  return pass_with_perf(std::move(sink));
}

// Emit pre-collected latency samples (us/op) under `metric`; perf mode only. For
// fixed-count functional loops (e.g. teardown-under-load) whose real iterations
// are timed in place rather than through a warmup+N PerfLoop.
inline void emit_latency_samples(PerfSink& sink, const std::string& metric,
                                 std::vector<double> samples_us) {
  if (perf_probe_mode() && !samples_us.empty()) {
    perf_detail::emit(sink, metric, 0, std::move(samples_us), "us/op");
  }
}

// Run `body` (a callable returning TestResult) with per-HIP_CHECK latency
// capture active -- a no-op with perf off -- and fold the captured latencies
// into the returned result. `body` owns exception handling so this header stays
// HIP-free. Used by both the in-process and isolated exec paths.
template <class Body>
inline TestResult run_with_probe(Body&& body) {
  PerfSink probe_sink;
  TestResult result;
  {
    PerfProbeScope probe(probe_sink);
    result = body();
  }

  for (auto& kv : probe_sink) {
    result.perf.push_back(std::move(kv));
  }
  return result;
}

}  // namespace acceptance
