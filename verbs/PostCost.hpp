/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// PostCost -- the pure (no ibverbs, no device I/O) helpers behind the ionic SQ
// post-cost test (verbs/post_cost.cc), kept in a header so
// tests/unit_postcost.cc can pin them on a machine with no NIC:
//
//   * summarize()          -- median, its 95% confidence interval, p99, ... of
//                             the per-call ibv_post_send cost
//   * clearly_slower()     -- "A is slower than B by more than the tolerance",
//                             judged on A's CI so noise cannot trip it
//   * above_reference()    -- the same against a hardcoded healthy figure, for
//                             hosts where the alternatives cannot be measured
//   * parse_positive()     -- the ACCEPT_POST_SEND_* environment overrides
//   * sum_mapped_kb()      -- KiB mapped by the uverbs device, from /proc/<pid>/maps
//
// The samples are per-call costs averaged over a block of back-to-back posts (see
// verbs/post_cost.cc for why single-call timestamps under-read an uncached
// MMIO path).
//
// Why the MEDIAN (and its CI) is compared: one ibv_post_send costs ~0.15-0.5 us
// when the send queue lives in host memory or in write-combining CMB and ~3 us
// when it lives in an UNCACHED controller memory buffer, because every byte of
// the WQE copy is then a posted MMIO store. That is a steady-state property of
// every call, not a tail event, so the median separates the regimes by ~10x while
// staying immune to the occasional preemption or page fault that would spoil a
// max-based comparison. The p99 is reported for diagnosis only.

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <istream>
#include <string>
#include <vector>

#include "common/Perf.hpp"

namespace acceptance {
namespace postcost {

// Default tolerance: a non-default SQ placement must beat the out-of-box one by
// more than this fraction before the out-of-box one counts as broken.
constexpr double kDefaultTolerancePct = 30.0;

// Hardcoded "healthy out-of-box" median ibv_post_send cost (ns) used when no
// alternative placement can be measured. This is the ionic figure, and also the
// generic fallback for NIC families without a measured one (e.g. ConnectX). Measured on TW MI350X / ionic
// 26.07.9.001: host-memory SQ 0.15-0.26 us, write-combining CMB 0.30-0.48 us per
// call, uncached CMB 2.9-3.0 us. 500 ns is the top of the healthy range; with the
// default 30% tolerance the verdict line sits at 650 ns, 4.5x below the broken
// mapping. Override with ACCEPT_POST_SEND_REF_NS.
constexpr double kDefaultReferenceNs = 500.0;
// Broadcom Thor (bnxt_re), expected healthy median: measured 155-167 ns per call
// on Ruby (MI350X, kernel 6.9.0 brcmrdma24, 3 runs, all five shapes) with no SQ
// placement choice; 250 ns leaves ~1.5x headroom for slower hosts, and with the
// default 30% tolerance the verdict line sits at 325 ns.
constexpr double kThorReferenceNs = 250.0;

// A uverbs mapping per QP larger than this means the SQ was placed in the CMB
// (a CMB SQ adds one queue-sized mapping per QP; a host-memory SQ adds none).
constexpr double kCmbMapKbPerQp = 4.0;

struct Stats {
  std::size_t n = 0;
  double      min = 0;
  double      median = 0;
  double      ci_lo = 0;  // 95% CI of the median (distribution-free, order statistics)
  double      ci_hi = 0;
  double      mean = 0;
  double      p99 = 0;
  double      max = 0;
};

// Summarise per-call costs (any unit; the result keeps it). Empty input -> zeros.
inline Stats summarize(std::vector<double> samples) {
  Stats s;
  if (samples.empty()) return s;

  s.n = samples.size();
  double total = 0;
  for (double v : samples) total += v;
  s.mean = total / static_cast<double>(s.n);
  s.median = perf_detail::quantile(samples, 0.50);  // sorts `samples`
  s.p99 = perf_detail::quantile(samples, 0.99);
  s.min = samples.front();
  s.max = samples.back();

  // Order-statistic CI of the median: ranks floor((n-1.96*sqrt(n))/2) and
  // ceil(1+(n+1.96*sqrt(n))/2), 1-based, clamped to the sample.
  const double n = static_cast<double>(s.n);
  const double half = 1.96 * std::sqrt(n);
  const double lo = std::max(1.0, std::floor((n - half) / 2.0));
  const double hi = std::min(n, std::ceil(1.0 + (n + half) / 2.0));
  s.ci_lo = samples[static_cast<std::size_t>(lo) - 1];
  s.ci_hi = samples[static_cast<std::size_t>(hi) - 1];
  return s;
}

// Parse a positive number from the raw env string. A missing, empty, non-numeric,
// non-finite, non-positive or > `max` value falls back (a typo must not silently
// turn the check off by parsing as 0 or a huge number).
inline double parse_positive(const char* env, double fallback, double max = 1e12) {
  if (env == nullptr || *env == '\0') return fallback;
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(env, &end);
  if (end == env || *end != '\0' || errno != 0) return fallback;
  if (!(v > 0.0) || v > max) return fallback;  // also rejects NaN
  return v;
}

// ACCEPT_POST_SEND_TOLERANCE_PCT as a fraction (30 -> 0.30).
inline double tolerance_frac() {
  return parse_positive(std::getenv("ACCEPT_POST_SEND_TOLERANCE_PCT"),
                        kDefaultTolerancePct, 100000.0) / 100.0;
}

// Expected healthy median for a NIC family ("ionic", "thor", "connectx", ...);
// ACCEPT_POST_SEND_REF_NS overrides it for every family.
inline double reference_ns_for(const std::string& family) {
  double ref = kDefaultReferenceNs;
  if (family == "thor") ref = kThorReferenceNs;
  return parse_positive(std::getenv("ACCEPT_POST_SEND_REF_NS"), ref);
}

// True when `slow` is slower than `base` by more than `tol` (a fraction), with
// confidence: even the low end of slow's median CI is above base's median grown
// by the tolerance. No samples on either side is never "slower".
inline bool clearly_slower(const Stats& slow, const Stats& base, double tol) {
  if (slow.n == 0 || base.n == 0) return false;
  return slow.ci_lo > base.median * (1.0 + tol);
}

// True when the low end of `s`'s median CI is above the hardcoded healthy
// reference grown by the tolerance.
inline bool above_reference(const Stats& s, double ref_ns, double tol) {
  if (s.n == 0) return false;
  return s.ci_lo > ref_ns * (1.0 + tol);
}

// Sum the size (KiB) of every mapping in a /proc/<pid>/maps stream whose line
// contains `needle` (the uverbs device path, "uverbs"). A line looks like
//   7f1c2a000000-7f1c2a010000 rw-s 00000000 00:06 123  /dev/infiniband/uverbs0
inline long sum_mapped_kb(std::istream& maps, const std::string& needle) {
  long total_kb = 0;
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(needle) == std::string::npos) continue;
    const std::size_t dash = line.find('-');
    const std::size_t space = line.find(' ');
    if (dash == std::string::npos || space == std::string::npos || dash > space) {
      continue;
    }
    char* end = nullptr;
    const unsigned long long lo = std::strtoull(line.c_str(), &end, 16);
    if (end != line.c_str() + dash) continue;
    const unsigned long long hi = std::strtoull(line.c_str() + dash + 1, &end, 16);
    if (end != line.c_str() + space || hi <= lo) continue;
    total_kb += static_cast<long>((hi - lo) / 1024);
  }
  return total_kb;
}

}  // namespace postcost
}  // namespace acceptance
