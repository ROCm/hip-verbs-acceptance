/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_policy — tests for the Policy CSV reader + classify() (common/Policy.*).
// Writes a temp policy/<vendor>.csv to a scratch dir, then exercises:
//   * exact-id beats wildcard; longest-literal-prefix wildcard wins, tie ->
//     first-declared; unlisted -> Unclassified; unknown vendor -> all Unclassified;
//   * expected_status parse (empty stays "");
//   * parse robustness: comments, blank lines, CRLF, leading/trailing spaces on
//     the first 5 fields, a reason containing commas (verbatim), a too-few-commas
//     row (no throw);
//   * load(): missing <vendor>.csv with the DEFAULT dir -> discovery (present=false,
//     no throw); explicitly-missing dir/file -> throws.
//   * eval_when — rocm/driver/fw/gpu/sym/!sym terms; version comparison
//     edge cases; multi-term AND; piecewise pattern; back-compat 2-arg classify.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include <unistd.h>
#include <sys/stat.h>

#include "common/Capability.hpp"
#include "common/Policy.hpp"
#include "common/Types.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

namespace accept_test {

namespace {

// Write `content` to <dir>/<vendor>.csv.
void write_csv(const std::string& dir, const std::string& vendor,
               const std::string& content) {
  std::ofstream f(dir + "/" + vendor + ".csv", std::ios::binary | std::ios::trunc);
  f << content;
  f.flush();
}

// Load helper: builds the file path from dir+vendor (mirrors the old 3-arg API).
Policy load_csv(const std::string& dir, const std::string& vendor,
                bool explicit_flag = true) {
  return Policy::load(dir + "/" + vendor + ".csv", explicit_flag);
}

const char* cn(Class c) { return to_string(c); }

// Check that test `id` classifies as `want`. `found` reflects whether the id
// was listed in the policy at all (false -> not run).
void check_class(const Policy& p, const std::string& /*vendor*/,
                 const std::string& id, Class want, const char* msg,
                 const EnvironmentInfo& env = EnvironmentInfo{}) {
  bool found = false;
  const Classification c = p.classify(id, env, &found);
  ++g_checks;
  // If the test is truly unlisted, found=false and cls is the zero-init Na.
  // Tests that check "unlisted -> not found" should use check_unlisted() below.
  if (c.cls != want) {
    ++g_failures;
    std::fprintf(stderr, "FAIL classify %s: id=%s got=%s want=%s (found=%d)\n",
                 msg, id.c_str(), cn(c.cls), cn(want), found);
  }
}

// Check that `id` is not listed in the policy (unlisted = not run).
void check_unlisted(const Policy& p, const std::string& id, const char* msg) {
  bool found = false;
  p.classify(id, EnvironmentInfo{}, &found);
  ++g_checks;
  if (found) {
    ++g_failures;
    std::fprintf(stderr, "FAIL unlisted %s: id=%s is listed (expected not found)\n",
                 msg, id.c_str());
  }
}

void test_classify_precedence(const std::string& dir) {
  // Declaration order matters for tie-breaks. Two wildcards with the SAME literal
  // prefix length ("syn_a_*" and "syn_a_?") test the first-declared tie-break.
  // "syn_*" (prefix 4) vs "syn_known_*" (prefix 10) test longest-prefix-wins.
  // CSV format: test,class,when,expected,ticket,reason (5-comma, when empty).
  const std::string csv =
      "syn_exact,MUST_PASS,,,TICK,exact wins over wildcard\n"
      "syn_*,NA,,,,short prefix wildcard\n"
      "syn_known_*,MUST_PASS,,,,longer prefix wildcard\n"
      "syn_a_*,KNOWN_FAILURE,,fail,,tie A first declared\n"
      "syn_a_?,MUST_PASS,,,,tie B same prefix len\n";
  write_csv(dir, "vtest", csv);
  const Policy p = load_csv(dir,"vtest");
  CHECK(p.present, "policy present after load");

  // Exact id beats any wildcard ("syn_exact" also matches "syn_*").
  check_class(p, "vtest", "syn_exact", Class::MustPass, "exact beats wildcard");

  // Longest literal prefix wins: "syn_known_x" matches both "syn_*" (prefix 4)
  // and "syn_known_*" (prefix 10) -> the longer one (MUST_PASS, not the short
  // wildcard's NA).
  check_class(p, "vtest", "syn_known_x", Class::MustPass,
              "longest literal prefix wins");

  // "syn_zzz" matches only "syn_*" -> NA.
  check_class(p, "vtest", "syn_zzz", Class::Na, "only short-prefix matches");

  // Tie on literal-prefix length: "syn_a_x" matches "syn_a_*" (len 6) AND
  // "syn_a_?" (len 6) -> first-declared wins (KNOWN_FAILURE).
  check_class(p, "vtest", "syn_a_x", Class::KnownFailure,
              "tie -> first declared");

  // Unlisted id (no match at all) -> not found (not run).
  check_unlisted(p, "totally_unlisted", "unlisted -> not found");

  // Unknown vendor (no such CSV, default dir) -> discovery mode (present=false).
  // In discovery mode classify returns MUST_PASS for all tests (found=true).
  const Policy none = load_csv(dir, "no_such_vendor", false);
  CHECK(!none.present, "unknown vendor (default dir) -> not present (discovery)");
  check_class(none, "no_such_vendor", "syn_exact", Class::MustPass,
              "discovery: all tests treated as MUST_PASS");
}

void test_expected_status(const std::string& dir) {
  // 6-field format: test,class,when,expected,ticket,reason
  const std::string csv =
      "kf_default,KNOWN_FAILURE,,,,no expected -> empty\n"
      "kf_pass,KNOWN_FAILURE,,pass,,expected pass\n"
      // Regression for the 6-field column-shift bug: a real-world KNOWN_FAILURE
      // row with empty `when` + expected=fail + ticket + reason must parse each
      // field into its own column AND still classify as KNOWN_FAILURE (the empty
      // `when` must not let `fail` leak into the predicate slot and flip it NA).
      "kf_real,KNOWN_FAILURE,,fail,ROCM-25387,deadlock, with a comma\n";
  write_csv(dir, "vexp", csv);
  const Policy p = load_csv(dir,"vexp");

  const Classification d = p.classify("kf_default", EnvironmentInfo{});
  CHECK_EQ_STR(d.expected_status, "",
               "empty expected_status parses as empty (Reporter defaults fail)");

  const Classification e = p.classify("kf_pass", EnvironmentInfo{});
  CHECK_EQ_STR(e.expected_status, "pass", "expected_status=pass parsed");

  const Classification r = p.classify("kf_real", EnvironmentInfo{});
  CHECK(r.cls == Class::KnownFailure,
        "KNOWN_FAILURE,,fail row classifies as KNOWN_FAILURE (no column shift)");
  CHECK_EQ_STR(r.expected_status, "fail", "expected_status=fail in column 4");
  CHECK_EQ_STR(r.ticket, "ROCM-25387", "ticket in column 5");
  CHECK_EQ_STR(r.reason, "deadlock, with a comma",
               "reason is verbatim remainder (comma preserved)");
}

void test_parse_robustness(const std::string& dir) {
  // # comments, blank lines, CRLF line endings, leading/trailing spaces on the
  // first 5 fields, a reason containing commas (verbatim), and a too-few-commas
  // row (must not throw, must still resolve a class).
  // Format: test,class,when,expected,ticket,reason (5 commas).
  const std::string csv =
      "# this is a comment\r\n"
      "\r\n"
      "  spaced_id  ,  must_pass  ,  ,  ,  TICK-9  ,  reason, with, commas  \r\n"
      "\n"
      "fewcommas,KNOWN_FAILURE\n";  // only 1 comma: test + class, rest empty
  write_csv(dir, "vrob", csv);

  // Must not throw on the malformed/odd rows.
  Policy p;
  bool threw = false;
  try {
    p = load_csv(dir,"vrob");
  } catch (...) {
    threw = true;
  }
  CHECK(!threw, "load tolerates comments/blank/CRLF/spaces/too-few-commas");

  // Leading/trailing spaces on the first 5 fields are trimmed: the id matches.
  const Classification s = p.classify("spaced_id", EnvironmentInfo{});
  CHECK(s.cls == Class::MustPass,
        "spaced first-5 fields trimmed (id+class resolve)");
  CHECK_EQ_STR(s.ticket, "TICK-9", "ticket field trimmed");
  // The reason is the verbatim remainder after the 5th comma, commas preserved
  // (NOT trimmed of leading space since it is the free-form last field).
  CHECK_EQ_STR(s.reason, "  reason, with, commas  ",
               "reason verbatim incl. commas and surrounding spaces");

  // Too-few-commas row: class still resolves (KNOWN_FAILURE), no crash.
  const Classification f = p.classify("fewcommas", EnvironmentInfo{});
  CHECK(f.cls == Class::KnownFailure, "too-few-commas row resolves its class");
}

void test_load_modes(const std::string& dir) {
  // Default dir (explicit_flag=false) with a missing <vendor>.csv -> discovery,
  // present=false, NO throw.
  bool threw = false;
  Policy disc;
  try {
    disc = load_csv(dir,"absent_vendor", false);
  } catch (...) {
    threw = true;
  }
  CHECK(!threw, "missing default <vendor>.csv does not throw");
  CHECK(!disc.present, "missing default <vendor>.csv -> discovery (present=false)");

  // Explicitly-named dir that does NOT exist -> throws (typo guard).
  bool threw_dir = false;
  try {
    (void)Policy::load("/no/such/dir_xyz/vtest.csv", /*explicit_flag=*/true);
  } catch (const std::exception&) {
    threw_dir = true;
  }
  CHECK(threw_dir, "explicit missing DIR throws");

  // Explicitly-named EXISTING dir but a missing <vendor>.csv -> throws (typo
  // guard distinguishes 'dir present, file absent').
  bool threw_file = false;
  try {
    (void)load_csv(dir,"definitely_absent_vendor");
  } catch (const std::exception&) {
    threw_file = true;
  }
  CHECK(threw_file, "explicit dir present but missing <vendor>.csv throws");
}

// ---------------------------------------------------------------------------
// eval_when / when-predicate unit tests
// ---------------------------------------------------------------------------

// Build an EnvironmentInfo with the specified fields (all others default).
static EnvironmentInfo make_env(const std::string& rocm = "",
                                const std::string& driver = "",
                                int gpu = 0,
                                const std::string& fw = "",
                                const std::map<std::string, bool>& syms = {}) {
  EnvironmentInfo e;
  e.rocm_version   = rocm;
  e.driver_version = driver;
  e.gpu_count      = gpu;
  if (!fw.empty()) {
    RdmaDevice d;
    d.fw_ver = fw;
    e.rdma_devices.push_back(d);
  }
  e.symbols = syms;
  return e;
}

void test_when_version_comparison(const std::string& dir) {
  // Policy with `when` column (5-comma format). Three rows for the same test:
  //   Row 1: NA when rocm<7.0 (old stack)
  //   Row 2: MUST_PASS (catch-all)
  //
  // Also exercises driver, fw, and gpu dims.
  const std::string csv =
      // rocm dim tests
      "test_rocm,NA,rocm<7.0,,T-1,old rocm\n"
      "test_rocm,MUST_PASS,,,,\n"
      // driver dim
      "test_drv,NA,driver<6.0,,T-2,old driver\n"
      "test_drv,MUST_PASS,,,,\n"
      // fw dim
      "test_fw,NA,fw<100.0,,T-3,old fw\n"
      "test_fw,MUST_PASS,,,,\n"
      // gpu dim
      "test_gpu,NA,gpu<2,,,no peer on single GPU\n"
      "test_gpu,MUST_PASS,,,,\n";
  write_csv(dir, "vwhen", csv);
  const Policy p = load_csv(dir,"vwhen");

  // The 4th arg of check_class is a descriptive label; it documents each case.
  check_class(p, "vwhen", "test_rocm", Class::Na,
              "rocm<7.0 with 6.5 -> NA", make_env("6.5.0"));
  check_class(p, "vwhen", "test_rocm", Class::MustPass,
              "rocm<7.0 with 7.0.0 -> MUST_PASS", make_env("7.0.0"));
  check_class(p, "vwhen", "test_rocm", Class::MustPass,
              "rocm<7.0 with empty rocm -> MUST_PASS", make_env(""));

  check_class(p, "vwhen", "test_drv", Class::Na,
              "driver<6.0 with 5.0 -> NA", make_env("", "5.0.0"));
  check_class(p, "vwhen", "test_drv", Class::MustPass,
              "driver<6.0 with 6.0.0 -> MUST_PASS", make_env("", "6.0.0"));

  check_class(p, "vwhen", "test_fw", Class::Na,
              "fw<100.0 with 99.0 -> NA", make_env("", "", 0, "99.0"));
  check_class(p, "vwhen", "test_fw", Class::MustPass,
              "fw<100.0 with 233.2.104.0 -> MUST_PASS", make_env("", "", 0, "233.2.104.0"));
  check_class(p, "vwhen", "test_fw", Class::MustPass,
              "fw<100.0 with no rdma -> MUST_PASS", make_env("", "", 0, ""));

  check_class(p, "vwhen", "test_gpu", Class::Na,
              "gpu<2 with gpu_count=1 -> NA", make_env("", "", 1, ""));
  check_class(p, "vwhen", "test_gpu", Class::MustPass,
              "gpu<2 with gpu_count=8 -> MUST_PASS", make_env("", "", 8, ""));
}

void test_when_version_edge_cases(const std::string& dir) {
  // Test edge cases of version comparison logic:
  //   7.0.1 >= 7.0  -> true
  //   7.12 > 7.0.2  -> true
  //   6.14.14 < 7.0 -> true
  //   3.104.0 >= 3.104.0 -> true (equality)
  //   7.0 >= 7.0.1  -> false (shorter pads with 0: 7.0.0 < 7.0.1)
  const std::string csv =
      "edge1,MUST_PASS,rocm>=7.0,,, 7.0.1 >= 7.0\n"
      "edge2,MUST_PASS,rocm>7.0.2,,, 7.12 > 7.0.2\n"
      "edge3,MUST_PASS,rocm<7.0,,, 6.14.14 < 7.0\n"
      "edge4,MUST_PASS,rocm>=3.104.0,,, equality\n"
      "edge5,MUST_PASS,rocm>=7.0.1,,, 7.0 >= 7.0.1 should be false\n";
  write_csv(dir, "vedge", csv);
  const Policy p = load_csv(dir,"vedge");

  check_class(p, "vedge", "edge1", Class::MustPass, "7.0.1 >= 7.0 -> true",
              make_env("7.0.1"));
  check_class(p, "vedge", "edge2", Class::MustPass, "7.12 > 7.0.2 -> true",
              make_env("7.12"));
  check_class(p, "vedge", "edge3", Class::MustPass, "6.14.14 < 7.0 -> true",
              make_env("6.14.14"));
  check_class(p, "vedge", "edge4", Class::MustPass, "3.104.0 >= 3.104.0 -> true",
              make_env("3.104.0"));
  // 7.0 >= 7.0.1: predicate false -> SKIP (software-gated).
  check_class(p, "vedge", "edge5", Class::Skip,
              "7.0 >= 7.0.1 -> false -> SKIP (software-gated predicate failed)",
              make_env("7.0"));
}

void test_when_sym(const std::string& dir) {
  const std::string csv =
      "sym_test,NA,!sym:libionic,,T-1,requires libionic\n"
      "sym_test,MUST_PASS,,,,\n"
      "sym2_test,NA,sym:some_cap,,,\n"
      "sym2_test,MUST_PASS,,,,\n";
  write_csv(dir, "vsym", csv);
  const Policy p = load_csv(dir,"vsym");

  check_class(p, "vsym", "sym_test", Class::Na, "!sym:libionic absent -> NA",
              make_env("", "", 0, "", {}));
  check_class(p, "vsym", "sym_test", Class::MustPass,
              "!sym:libionic present -> MUST_PASS",
              make_env("", "", 0, "", {{"libionic", true}}));
  check_class(p, "vsym", "sym_test", Class::Na, "!sym:libionic false -> NA",
              make_env("", "", 0, "", {{"libionic", false}}));

  check_class(p, "vsym", "sym2_test", Class::Na, "sym:some_cap present -> NA",
              make_env("", "", 0, "", {{"some_cap", true}}));
  check_class(p, "vsym", "sym2_test", Class::MustPass,
              "sym:some_cap absent -> MUST_PASS (catch-all)", make_env());
}

void test_when_multi_term_and(const std::string& dir) {
  // Multi-term AND: both must hold.
  const std::string csv =
      "multi,KNOWN_FAILURE,driver<6.0;rocm<7.0,fail,T-1,old stack\n"
      "multi,MUST_PASS,,,,\n";
  write_csv(dir, "vmulti", csv);
  const Policy p = load_csv(dir,"vmulti");

  check_class(p, "vmulti", "multi", Class::KnownFailure,
              "driver<6.0;rocm<7.0 both old -> KNOWN_FAILURE",
              make_env("6.0.0", "5.0.0"));
  check_class(p, "vmulti", "multi", Class::MustPass,
              "driver<6.0;rocm<7.0 rocm new -> MUST_PASS",
              make_env("7.0.0", "5.0.0"));
  check_class(p, "vmulti", "multi", Class::MustPass,
              "driver<6.0;rocm<7.0 driver new -> MUST_PASS",
              make_env("6.0.0", "7.0.0"));
}

void test_when_multi_term_cross_dim(const std::string& dir) {
  // All four dims combined in multi-term AND predicates, including fw.
  // This exercises every dim in a compound condition, not just driver;rocm.
  const std::string csv =
      // rocm + fw: NA only when both old
      "rfw,NA,rocm<7.0;fw<100.0,,T-10,old rocm AND old fw\n"
      "rfw,MUST_PASS,,,,\n"
      // driver + fw: NA only when both old
      "dfw,NA,driver<6.0;fw<100.0,,T-11,old driver AND old fw\n"
      "dfw,MUST_PASS,,,,\n"
      // All four: rocm + driver + fw + gpu
      "all4,NA,rocm<7.0;driver<6.0;fw<100.0;gpu>=2,,T-12,all four dims\n"
      "all4,MUST_PASS,,,,\n";
  write_csv(dir, "vcross", csv);
  const Policy p = load_csv(dir,"vcross");

  const auto env_both_old = make_env("6.5.0", "5.0.0", 0, "99.0");
  const auto env_rocm_new = make_env("7.0.0", "5.0.0", 0, "99.0");
  const auto env_fw_new   = make_env("6.5.0", "5.0.0", 0, "233.0");

  check_class(p, "vcross", "rfw", Class::Na,
              "rocm<7.0;fw<100.0 both old -> NA", env_both_old);
  check_class(p, "vcross", "rfw", Class::MustPass,
              "rocm<7.0;fw<100.0 rocm new -> MUST_PASS", env_rocm_new);
  check_class(p, "vcross", "rfw", Class::MustPass,
              "rocm<7.0;fw<100.0 fw new -> MUST_PASS", env_fw_new);

  check_class(p, "vcross", "dfw", Class::Na,
              "driver<6.0;fw<100.0 both old -> NA", env_both_old);
  const auto env_driver_new = make_env("6.5.0", "6.0.0", 0, "99.0");
  check_class(p, "vcross", "dfw", Class::MustPass,
              "driver<6.0;fw<100.0 driver new -> MUST_PASS", env_driver_new);

  const auto env_all_match = make_env("6.5.0", "5.0.0", 8, "99.0");
  check_class(p, "vcross", "all4", Class::Na,
              "all four dims hold -> NA", env_all_match);
  const auto env_nogpu = make_env("6.5.0", "5.0.0", 1, "99.0");
  check_class(p, "vcross", "all4", Class::MustPass,
              "all four: gpu>=2 fails (gpu=1) -> MUST_PASS", env_nogpu);
}

void test_when_piecewise(const std::string& dir) {
  // Piecewise pattern: first row has a failing predicate; second is catch-all.
  // Verifies that a failed predicate causes fallthrough, not Unclassified.
  const std::string csv =
      "piece,NA,rocm<7.0,,T-1,old stack\n"
      "piece,MUST_PASS,,,,\n";
  write_csv(dir, "vpiece", csv);
  const Policy p = load_csv(dir,"vpiece");

  check_class(p, "vpiece", "piece", Class::MustPass,
              "piecewise: predicate fails -> catch-all MUST_PASS",
              make_env("8.0.0"));
  check_class(p, "vpiece", "piece", Class::Na,
              "piecewise: predicate passes -> NA",
              make_env("6.5.0"));
}

void test_when_single_row_na_fallback(const std::string& dir) {
  // single-row pattern: `vat_foo,MUST_PASS,rocm>=7.12` works without a
  // catch-all NA row. When the predicate fails the test is known to the policy but
  // not applicable in this environment -> NA (not Unclassified).
  const std::string csv =
      "single,MUST_PASS,rocm>=7.12,,,\n";
  write_csv(dir, "vsingle", csv);
  const Policy p = load_csv(dir,"vsingle");

  check_class(p, "vsingle", "single", Class::MustPass,
              "single-row: rocm>=7.12 with 7.12.0 -> MUST_PASS",
              make_env("7.12.0"));
  check_class(p, "vsingle", "single", Class::Skip,
              "single-row: rocm>=7.12 with 7.0.0 -> SKIP (software-gated)",
              make_env("7.0.0"));
  check_class(p, "vsingle", "single", Class::Skip,
              "single-row: rocm>=7.12 with missing rocm -> SKIP (software dim, dim absent)",
              make_env(""));
  // Unlisted test -> not found (not run, distinct from in-policy NA/SKIP).
  check_unlisted(p, "unlisted", "single-row: unlisted test -> not found");
}

void test_when_exact_shadows_glob(const std::string& dir) {
  // Precedence foot-gun pin: an exact-id row that MATCHES by id takes precedence
  // over globs even when its `when` predicate FAILS. The exhausted exact bucket
  // resolves to NA (in-policy-but-not-applicable) and never falls through to a
  // glob catch-all. Policy authors must put catch-all behavior in exact rows.
  const std::string csv =
      "foo_x,MUST_PASS,rocm>=9.9,,T-1,needs ROCm 9.9 (will fail on 7.0)\n"
      "foo_*,KNOWN_FAILURE,,fail,,glob catch-all that is deliberately NOT reached\n";
  write_csv(dir, "vshadow", csv);
  const Policy p = load_csv(dir,"vshadow");

  // Exact predicate fails (rocm 7.0 < 9.9); software dim -> SKIP, not the glob's KNOWN_FAILURE.
  check_class(p, "vshadow", "foo_x", Class::Skip,
              "exact-fails shadows glob -> SKIP (software predicate, not glob KNOWN_FAILURE)",
              make_env("7.0.0"));
  // A different id only the glob matches -> KNOWN_FAILURE (glob still works for it).
  check_class(p, "vshadow", "foo_y", Class::KnownFailure,
              "glob applies to ids with no exact row", make_env("7.0.0"));
}

void test_when_backcompat(const std::string& dir) {
  // Back-compat: 2-arg classify uses default EnvironmentInfo (all empty) ->
  // only catch-all rows (empty `when`) match.
  const std::string csv =
      "bc_test,NA,rocm<7.0,,T-1,only on old rocm\n"
      "bc_test,MUST_PASS,,,,\n";
  write_csv(dir, "vbc", csv);
  const Policy p = load_csv(dir,"vbc");

  // 2-arg classify: rocm_version is empty -> rocm<7.0 predicate false ->
  // catch-all MUST_PASS.
  const Classification c = p.classify("bc_test", EnvironmentInfo{});
  ++g_checks;
  if (c.cls != Class::MustPass) {
    ++g_failures;
    std::fprintf(stderr,
                 "FAIL backcompat 2-arg classify: got=%s want=must_pass\n",
                 to_string(c.cls));
  }
}

void test_when_empty_env_all_fail(const std::string& dir) {
  // All non-empty `when` predicates evaluate to false when EnvironmentInfo is
  // default-constructed (all fields empty). Only catch-all (empty `when`) rows
  // match.
  const std::string csv =
      "e1,NA,rocm<99.0,,,\n"
      "e2,NA,driver<99.0,,,\n"
      "e3,NA,fw<999.0,,,\n"
      "e4,NA,gpu<999,,,\n"
      "e5,NA,sym:any_sym,,,\n"
      // !sym: absent sym -> true even with empty env (symbols map is empty,
      // so the symbol is absent -> !sym is true).
      "e6,NA,!sym:missing_sym,,,\n";
  write_csv(dir, "vempty", csv);
  const Policy p = load_csv(dir,"vempty");
  const EnvironmentInfo empty_env;

  // Software-gated dims with empty env: predicate false -> SKIP.
  check_class(p, "vempty", "e1", Class::Skip,
              "empty env: rocm<99 -> false -> SKIP (software dim)", empty_env);
  check_class(p, "vempty", "e2", Class::Skip,
              "empty env: driver<99 -> false -> SKIP (software dim)", empty_env);
  check_class(p, "vempty", "e3", Class::Skip,
              "empty env: fw<999 -> false -> SKIP (software dim)", empty_env);
  // gpu dim: gpu_count==0, gpu<999 -> 0<999 is true -> row fires -> class is NA.
  check_class(p, "vempty", "e4", Class::Na,
              "empty env: gpu<999 with gpu_count=0 -> true -> NA (row fires)", empty_env);
  // sym:any_sym: symbol absent -> predicate false -> SKIP (software dim).
  check_class(p, "vempty", "e5", Class::Skip,
              "empty env: sym:any_sym absent -> false -> SKIP (software dim)", empty_env);
  // !sym:missing_sym: symbol absent -> !sym is true -> row fires -> NA.
  check_class(p, "vempty", "e6", Class::Na,
              "empty env: !sym:missing_sym absent -> true -> NA (row fires)", empty_env);
}

void test_when_matched_field(const std::string& dir) {
  // Verify that when_matched is set correctly in the returned Classification.
  const std::string csv =
      "wm_test,NA,rocm<7.0,,T-1,old rocm\n"
      "wm_test,MUST_PASS,,,,\n";
  write_csv(dir, "vwm", csv);
  const Policy p = load_csv(dir,"vwm");

  // When predicate fires: when_matched == "rocm<7.0"
  {
    const Classification c =
        p.classify("wm_test", make_env("6.0.0"));
    ++g_checks;
    if (c.when_matched != "rocm<7.0") {
      ++g_failures;
      std::fprintf(stderr,
                   "FAIL when_matched (predicate fires): got='%s' want='rocm<7.0'\n",
                   c.when_matched.c_str());
    }
  }
  // When catch-all fires: when_matched == "" (empty)
  {
    const Classification c =
        p.classify("wm_test", make_env("8.0.0"));
    ++g_checks;
    if (!c.when_matched.empty()) {
      ++g_failures;
      std::fprintf(stderr,
                   "FAIL when_matched (catch-all): got='%s' want=''\n",
                   c.when_matched.c_str());
    }
  }
}

}  // namespace

void run_policy_tests() {
  // Scratch dir for the temp CSV files.
  char tmpl[] = "/tmp/accept_unit_policy_XXXXXX";
  const char* dir = ::mkdtemp(tmpl);
  if (dir == nullptr) {
    ++g_failures;
    ++g_checks;
    std::fprintf(stderr, "FAIL: mkdtemp for policy unit tests\n");
    return;
  }
  const std::string d = dir;

  test_classify_precedence(d);
  test_expected_status(d);
  test_parse_robustness(d);
  test_load_modes(d);
  // when/predicate tests.
  test_when_version_comparison(d);
  test_when_version_edge_cases(d);
  test_when_sym(d);
  test_when_multi_term_and(d);
  test_when_multi_term_cross_dim(d);
  test_when_single_row_na_fallback(d);
  test_when_exact_shadows_glob(d);
  test_when_piecewise(d);
  test_when_backcompat(d);
  test_when_empty_env_all_fail(d);
  test_when_matched_field(d);

  // Best-effort cleanup of the scratch CSVs + dir.
  for (const char* v : {"vtest", "vexp", "vrob", "vwhen", "vedge", "vsym",
                         "vmulti", "vcross", "vpiece", "vsingle", "vshadow",
                         "vbc", "vempty", "vwm"}) {
    std::remove((d + "/" + v + ".csv").c_str());
  }
  ::rmdir(d.c_str());
}

}  // namespace accept_test
