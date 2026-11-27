/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_postcost -- the decision rules of the ibv_post_send cost test
// (verbs/post_cost.cc) pinned without a NIC: the median and its 95% CI, the
// ACCEPT_POST_SEND_* parsing, "default is clearly slower than an alternative /
// than the hardcoded reference" and the /proc/<pid>/maps accounting that tells a
// CMB-resident SQ from a host one.

#include <sstream>
#include <string>
#include <vector>

#include "tests/unit_harness.hpp"
#include "verbs/PostCost.hpp"

namespace accept_test {

namespace {

using namespace acceptance::postcost;

// `n` samples around `center` with +/- `spread` deterministic jitter.
std::vector<double> around(double center, double spread, int n) {
  std::vector<double> v;
  for (int i = 0; i < n; ++i)
    v.push_back(center + spread * ((i % 11) - 5) / 5.0);
  return v;
}

void test_summarize() {
  Stats s = summarize({});
  CHECK(s.n == 0 && s.median == 0.0, "empty input summarizes to zeros");

  s = summarize({5.0, 1.0, 3.0});
  CHECK(s.n == 3, "sample count");
  CHECK(s.min == 1.0 && s.max == 5.0, "min/max of unsorted input");
  CHECK(s.median == 3.0, "median of an odd sample set");
  CHECK(s.mean == 3.0, "mean");
  CHECK(s.ci_lo >= 1.0 && s.ci_hi <= 5.0 && s.ci_lo <= s.median &&
            s.median <= s.ci_hi,
        "CI of a tiny sample stays inside the sample and brackets the median");

  s = summarize({10.0, 20.0});
  CHECK(s.median == 15.0, "median of an even sample set interpolates");

  // 5% slow outliers do not move the median; p99 does see them.
  std::vector<double> v(190, 150.0);
  for (int i = 0; i < 10; ++i) v.push_back(40000.0);
  s = summarize(v);
  CHECK(s.median == 150.0, "outliers do not move the median");
  CHECK(s.ci_hi == 150.0, "outliers do not widen the median CI");
  CHECK(s.p99 > 1000.0, "p99 does see the outliers (diagnosis only)");

  // A noisy sample gets a CI that brackets the median and is not degenerate.
  s = summarize(around(1000.0, 100.0, 512));
  CHECK(s.ci_lo <= s.median && s.median <= s.ci_hi, "noisy CI brackets the median");
  CHECK(s.ci_lo >= s.min && s.ci_hi <= s.max, "CI stays inside the sample");
  CHECK(s.ci_hi - s.ci_lo < 100.0, "CI of 512 samples is tighter than the spread");
}

void test_parse_positive() {
  CHECK(parse_positive(nullptr, 7.0) == 7.0, "unset -> fallback");
  CHECK(parse_positive("", 7.0) == 7.0, "empty -> fallback");
  CHECK(parse_positive("2500", 7.0) == 2500.0, "plain number");
  CHECK(parse_positive("750.5", 7.0) == 750.5, "fractional number");
  // A typo must never turn the check off (0 / garbage / negative / absurd).
  CHECK(parse_positive("0", 7.0) == 7.0, "zero is rejected");
  CHECK(parse_positive("-5", 7.0) == 7.0, "negative is rejected");
  CHECK(parse_positive("abc", 7.0) == 7.0, "non-numeric is rejected");
  CHECK(parse_positive("12us", 7.0) == 7.0, "trailing junk is rejected");
  CHECK(parse_positive("nan", 7.0) == 7.0, "nan is rejected");
  CHECK(parse_positive("inf", 7.0) == 7.0, "inf is rejected");
  CHECK(parse_positive("1e30", 7.0) == 7.0, "absurdly large is rejected");
  CHECK(parse_positive("500", 7.0, 100.0) == 7.0, "above the caller's max is rejected");
  CHECK(kDefaultTolerancePct == 30.0, "default tolerance is 30%");
}

void test_clearly_slower() {
  const double tol = 0.30;
  const Stats uc = summarize(around(2950.0, 30.0, 512));     // uncached CMB
  const Stats host = summarize(around(150.0, 10.0, 512));    // host memory
  const Stats wc = summarize(around(300.0, 20.0, 512));      // write-combining CMB

  CHECK(clearly_slower(uc, host, tol), "uncached default is slower than host SQ");
  CHECK(clearly_slower(uc, wc, tol), "uncached default is slower than WC CMB");
  CHECK(!clearly_slower(host, uc, tol), "host SQ is not slower than uncached");

  // Within the tolerance is not flagged: 1.2x with a 30% tolerance.
  const Stats a = summarize(around(1200.0, 20.0, 512));
  const Stats b = summarize(around(1000.0, 20.0, 512));
  CHECK(!clearly_slower(a, b, tol), "1.2x slower is inside a 30% tolerance");
  CHECK(clearly_slower(a, b, 0.10), "1.2x slower is outside a 10% tolerance");

  // Noise must not flag: overlapping distributions with a 1.1x median gap.
  const Stats n1 = summarize(around(1100.0, 400.0, 512));
  const Stats n2 = summarize(around(1000.0, 400.0, 512));
  CHECK(!clearly_slower(n1, n2, tol), "overlapping noisy samples are not flagged");

  CHECK(!clearly_slower(Stats{}, host, tol), "no samples is never slower");
  CHECK(!clearly_slower(uc, Stats{}, tol), "no baseline is never faster");
}

void test_reference_for() {
  // (environment overrides are not set under the unit runner)
  CHECK(reference_ns_for("ionic") == kDefaultReferenceNs, "ionic reference");
  CHECK(reference_ns_for("thor") == kThorReferenceNs, "thor reference");
  CHECK(reference_ns_for("thor") < reference_ns_for("ionic"),
        "thor expects a cheaper post than the ionic ceiling");
  CHECK(reference_ns_for("connectx") == kDefaultReferenceNs,
        "unmeasured families use the generic reference");
  // Thor's measured 155-170 ns passes, an uncached-MMIO-like 3 us does not.
  CHECK(!above_reference(summarize(around(165.0, 5.0, 512)), kThorReferenceNs, 0.30),
        "measured Thor post is under its reference");
  CHECK(above_reference(summarize(around(3000.0, 30.0, 512)), kThorReferenceNs, 0.30),
        "a 3 us post is flagged on Thor");
}

void test_above_reference() {
  const double tol = 0.30;
  CHECK(above_reference(summarize(around(2950.0, 30.0, 512)), kDefaultReferenceNs, tol),
        "uncached post is above the hardcoded reference");
  CHECK(!above_reference(summarize(around(150.0, 10.0, 512)), kDefaultReferenceNs, tol),
        "host-memory post is under the reference");
  CHECK(!above_reference(summarize(around(480.0, 20.0, 512)), kDefaultReferenceNs, tol),
        "write-combining post (top of healthy range) is under the reference");
  // Line sits at reference * (1 + tolerance) = 650 ns.
  CHECK(!above_reference(summarize(around(640.0, 5.0, 512)), kDefaultReferenceNs, tol),
        "just under the tolerance line is fine");
  CHECK(above_reference(summarize(around(700.0, 5.0, 512)), kDefaultReferenceNs, tol),
        "clearly over the tolerance line is flagged");
  CHECK(!above_reference(Stats{}, kDefaultReferenceNs, tol), "no samples is not flagged");
}

void test_sum_mapped_kb() {
  // 64 KiB (a CMB SQ) + 8 KiB (context) of uverbs mappings; the anonymous and
  // [heap] mappings and the malformed uverbs line must not count.
  std::istringstream in(
      "7f1c2a000000-7f1c2a010000 rw-s 00000000 00:06 123  /dev/infiniband/uverbs0\n"
      "7f1c2b000000-7f1c2b002000 rw-s 00000000 00:06 123  /dev/infiniband/uverbs0\n"
      "7f1c2c000000-7f1c2c100000 rw-p 00000000 00:00 0\n"
      "55d0c0000000-55d0c0021000 rw-p 00000000 00:00 0    [heap]\n"
      "zz-yy rw-s 0 0 0 uverbs\n");
  CHECK(sum_mapped_kb(in, "uverbs") == 72, "sums only the well-formed uverbs mappings");
}

}  // namespace

void run_postcost_tests() {
  test_summarize();
  test_parse_positive();
  test_clearly_slower();
  test_above_reference();
  test_reference_for();
  test_sum_mapped_kb();
}

}  // namespace accept_test
