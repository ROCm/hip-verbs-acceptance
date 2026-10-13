/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/Policy.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fnmatch.h>

namespace acceptance {

namespace {

// Lowercase a copy of `s`. CSV files use UPPERCASE class tokens (MUST_PASS,
// KNOWN_FAILURE, NA, ...) for human readability, but class_from expects
// lowercase. We normalize here and accept either case.
std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Trim leading/trailing ASCII whitespace.
std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// True if `key` contains glob metacharacters (so it is a wildcard pattern).
bool is_glob(const std::string& key) {
  return key.find_first_of("*?[") != std::string::npos;
}

// Length of the literal prefix of a glob pattern (chars before the first
// metacharacter). Used to pick the most-specific wildcard on multiple matches.
std::size_t literal_prefix_len(const std::string& key) {
  const std::size_t p = key.find_first_of("*?[");
  return p == std::string::npos ? key.size() : p;
}

// True if every term in a `when` predicate is software-gated (rocm, driver,
// fw, sym, !sym). A failing software predicate resolves to SKIP
// instead of NA. A predicate containing `gpu` or an empty string (catch-all)
// is not purely software-gated and resolves to NA on failure.
bool is_software_predicate(const std::string& when_str) {
  if (when_str.empty()) return false;  // empty = catch-all, not software-gated
  // Split on ';' and check each term.
  std::size_t pos = 0;
  while (pos <= when_str.size()) {
    const std::size_t end = when_str.find(';', pos);
    const std::string term = trim(
        when_str.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
    pos = (end == std::string::npos) ? when_str.size() + 1 : end + 1;
    if (term.empty()) continue;
    // sym: and !sym: are software (library presence).
    if (term.size() >= 4 && term.substr(0, 4) == "sym:") continue;
    if (term.size() >= 5 && term.substr(0, 5) == "!sym:") continue;
    // rocm/driver/fw are software (stack version).
    // "fw" must be followed immediately by a comparison operator to avoid
    // matching unrelated tokens like "fwupd_version".
    const bool is_fw = term.size() >= 3 && term.substr(0, 2) == "fw" &&
                       (term[2] == '>' || term[2] == '<' || term[2] == '=');
    if (term.substr(0, 4) == "rocm" || term.substr(0, 6) == "driver" ||
        is_fw) continue;
    // Anything else (e.g. gpu count) is hardware-gated.
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// `when` predicate evaluator
// ---------------------------------------------------------------------------

// Split `s` on delimiter `delim`. Returns a vector of substrings (never empty:
// at least one element even for an empty input string).
std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (true) {
    const std::size_t pos = s.find(delim, start);
    if (pos == std::string::npos) {
      parts.push_back(s.substr(start));
      break;
    }
    parts.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

// Parse a version string like "7.0.1" or "6.14.14" into a vector of numeric
// components. Leading numeric dot-separated components only; non-numeric suffix
// and trailing components after the first non-numeric one are ignored.
// Returns empty vector on completely non-parseable input.
std::vector<unsigned long> parse_version(const std::string& ver) {
  std::vector<unsigned long> comps;
  if (ver.empty()) return comps;
  const std::vector<std::string> parts = split(ver, '.');
  for (const std::string& p : parts) {
    if (p.empty()) break;
    // Only the leading digits matter; stop at first non-digit.
    std::size_t i = 0;
    while (i < p.size() && std::isdigit(static_cast<unsigned char>(p[i]))) ++i;
    if (i == 0) break;  // non-numeric: stop
    // Parse with stoul; guard against stoul throwing on overflow.
    try {
      comps.push_back(std::stoul(p.substr(0, i)));
    } catch (...) {
      break;
    }
  }
  return comps;
}

// Compare two version component vectors lexicographically by numeric component,
// padding the shorter one with zeros.
// Returns -1, 0, or +1 like strcmp.
int cmp_version(const std::vector<unsigned long>& a,
                const std::vector<unsigned long>& b) {
  const std::size_t n = std::max(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned long av = i < a.size() ? a[i] : 0UL;
    const unsigned long bv = i < b.size() ? b[i] : 0UL;
    if (av < bv) return -1;
    if (av > bv) return  1;
  }
  return 0;
}

// Apply a comparison operator string (">=", "<=", ">", "<", "==", "!=") to an
// integer cmp result (-1/0/1). Returns false for unrecognized operators.
bool apply_op(int cmp, const std::string& op) {
  if (op == ">=") return cmp >= 0;
  if (op == "<=") return cmp <= 0;
  if (op == ">")  return cmp >  0;
  if (op == "<")  return cmp <  0;
  if (op == "==") return cmp == 0;
  if (op == "!=") return cmp != 0;
  return false;  // unknown op -> conservative false
}

// Evaluate a single `when` term against `env`. Returns false on any parse
// failure or when the relevant field is empty/missing.
//
// Term forms:
//   sym:<name>   — env.symbols["<name>"] is true
//   !sym:<name>  — env.symbols["<name>"] is false (or absent)
//   <dim><op><value>  — where dim in {rocm, driver, fw, gpu}
//                       and op in {>=, <=, >, <, ==, !=}
bool eval_term(const std::string& term, const EnvironmentInfo& env) {
  // sym:<name> / !sym:<name>
  if (term.size() >= 4 && term.substr(0, 4) == "sym:") {
    const std::string name = term.substr(4);
    auto it = env.symbols.find(name);
    return it != env.symbols.end() && it->second;
  }
  if (term.size() >= 5 && term.substr(0, 5) == "!sym:") {
    const std::string name = term.substr(5);
    auto it = env.symbols.find(name);
    return it == env.symbols.end() || !it->second;
  }

  // <dim><op><value>: parse dimension (rocm|driver|fw|gpu), then op, then value.
  // Ops are 1 or 2 chars; we try 2-char first.
  static const char* dims[] = {"rocm", "driver", "fw", "gpu", nullptr};
  for (int di = 0; dims[di] != nullptr; ++di) {
    const std::string dim = dims[di];
    if (term.size() <= dim.size()) continue;
    if (term.substr(0, dim.size()) != dim) continue;

    // After the dim name comes the operator.
    const std::string rest = term.substr(dim.size());
    std::string op;
    std::string val_str;

    // Try 2-char operators first, then 1-char.
    if (rest.size() >= 3 &&
        (rest.substr(0, 2) == ">=" || rest.substr(0, 2) == "<=" ||
         rest.substr(0, 2) == "==" || rest.substr(0, 2) == "!=")) {
      op      = rest.substr(0, 2);
      val_str = rest.substr(2);
    } else if (rest.size() >= 2 &&
               (rest[0] == '>' || rest[0] == '<')) {
      op      = rest.substr(0, 1);
      val_str = rest.substr(1);
    } else {
      return false;  // malformed term
    }

    if (val_str.empty()) return false;

    if (dim == "gpu") {
      // Integer comparison.
      unsigned long lhs_val = 0;
      unsigned long rhs_val = 0;
      try {
        lhs_val = static_cast<unsigned long>(env.gpu_count);
        rhs_val = std::stoul(val_str);
      } catch (...) {
        return false;
      }
      // gpu_count == 0 could be genuinely zero GPUs or "not detected". For the
      // purposes of missing-dim behavior, treat it as a valid value.
      const int c = (lhs_val < rhs_val) ? -1 : (lhs_val > rhs_val) ? 1 : 0;
      return apply_op(c, op);
    }

    // Version string comparison (rocm / driver / fw).
    std::string lhs_str;
    if (dim == "rocm")   lhs_str = env.rocm_version;
    else if (dim == "driver") lhs_str = env.driver_version;
    else if (dim == "fw") {
      if (env.rdma_devices.empty()) return false;  // no device -> missing
      // LIMITATION: only the first enumerated RDMA device's firmware version is
      // checked. On a multi-NIC node where devices run different firmware levels,
      // a `fw>=X` predicate may not reflect the version of the NIC actually used
      // for a given test. Homogeneous firmware across all NICs (the typical cluster
      // deployment) is assumed for now; a future fix would allow the test to
      // declare which NIC it uses or check the min/max across all devices.
      lhs_str = env.rdma_devices[0].fw_ver;
    }

    // Missing/empty field -> false (cannot verify, do not apply the row).
    if (lhs_str.empty()) return false;

    const std::vector<unsigned long> lhs = parse_version(lhs_str);
    const std::vector<unsigned long> rhs = parse_version(val_str);
    if (lhs.empty()) return false;  // unparseable lhs -> false
    // rhs empty means "0" effectively — but a non-parseable rhs is a bug in the
    // CSV; be conservative and return false.
    if (rhs.empty() && !val_str.empty() &&
        !std::isdigit(static_cast<unsigned char>(val_str[0]))) {
      return false;
    }
    return apply_op(cmp_version(lhs, rhs), op);
  }

  return false;  // no dim matched
}

// Evaluate a full `when` predicate (`;`-separated AND of terms).
// Empty predicate -> always true (catch-all).
bool eval_when(const std::string& when_str, const EnvironmentInfo& env) {
  if (when_str.empty()) return true;
  const std::vector<std::string> terms = split(when_str, ';');
  for (const std::string& term : terms) {
    const std::string t = trim(term);
    if (t.empty()) continue;  // tolerate trailing/doubled semicolons
    if (!eval_term(t, env)) return false;
  }
  return true;
}

}  // namespace

// Parse one non-comment CSV row into an Entry.
// Format: test,class,when,expected,ticket,reason  (6 fields, split on 5 commas)
// `reason` is the verbatim remainder and may contain commas. All other fields
// are whitespace-trimmed.
//
// Rows with fewer than 5 commas are parsed positionally from the left; missing
// trailing fields default to empty (when=always, expected="", ticket="", reason="").
// Rows with more than 5 commas have the extra commas absorbed into reason — correct
// for reason text that contains commas. A stray extra comma before reason shifts
// the ticket into reason and leaves ticket empty; policy authors should count commas.
Policy::Entry Policy::parse_row(const std::string& row) {
  constexpr std::size_t kFields = 6;          // test,class,when,expected,ticket,reason
  constexpr std::size_t kSplits = kFields - 1;  // commas to split on (reason = remainder)
  std::size_t pos[kSplits];
  std::size_t found = 0;
  std::size_t from = 0;
  while (found < kSplits) {
    const std::size_t c = row.find(',', from);
    if (c == std::string::npos) break;
    pos[found++] = c;
    from = c + 1;
  }

  auto field = [&](std::size_t start, std::size_t end) {
    return row.substr(start, end - start);
  };

  std::string f_test, f_class, f_when, f_expected, f_ticket, f_reason;
  if (found >= 1) f_test  = field(0, pos[0]);
  else            f_test  = row;  // no commas at all -> whole row is the test id

  if (found >= 1) {
    const std::size_t c2_start = pos[0] + 1;
    if (found >= 2) f_class = field(c2_start, pos[1]);
    else            f_class = field(c2_start, row.size());
  }
  if (found >= 2) {
    const std::size_t c3_start = pos[1] + 1;
    if (found >= 3) f_when = field(c3_start, pos[2]);
    else            f_when = field(c3_start, row.size());
  }
  if (found >= 3) {
    const std::size_t c4_start = pos[2] + 1;
    if (found >= 4) f_expected = field(c4_start, pos[3]);
    else            f_expected = field(c4_start, row.size());
  }
  if (found >= 4) {
    const std::size_t c5_start = pos[3] + 1;
    if (found >= 5) f_ticket = field(c5_start, pos[4]);
    else            f_ticket = field(c5_start, row.size());
  }
  if (found >= kSplits) {
    // reason = verbatim remainder after the last split comma (commas preserved).
    f_reason = row.substr(pos[kSplits - 1] + 1);
  }

  const std::string cls_token = to_lower(trim(f_class));
  if (!cls_token.empty() && !class_is_valid(cls_token) &&
      cls_token != "regression" && cls_token != "known_broken") {
    std::fprintf(stderr,
                 "policy WARNING: unknown class token '%s' in row '%s'; "
                 "test will be treated as NA (not run).\n",
                 cls_token.c_str(), trim(f_test).c_str());
  }

  Policy::Entry e;
  e.test                = trim(f_test);
  e.when                = trim(f_when);
  e.cls.cls             = class_from(cls_token);
  e.cls.expected_status = trim(f_expected);
  e.cls.ticket          = trim(f_ticket);
  e.cls.reason          = f_reason;  // verbatim (NOT trimmed: free-form last field)
  return e;
}

Policy Policy::load(const std::string& path, bool explicit_flag) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (explicit_flag) {
      throw std::runtime_error("policy file not found or unreadable: " + path);
    }
    // Not explicitly requested and file missing -> discovery mode (caller warns).
    return Policy{};  // present == false
  }

  Policy p;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF
    const std::string t = trim(line);
    if (t.empty()) continue;  // blank line
    if (t[0] == '#') continue;  // comment
    p.entries_.push_back(parse_row(line));
  }
  p.present = true;
  return p;
}

// Build the synthetic fallback classification when a test is in-policy but no
// predicate matched. If the failing predicate is software-gated
// (rocm/driver/fw/sym dims) the result is SKIP; otherwise NA.
// Uses the first non-matching row's ticket/reason intentionally: when multiple
// rows exist for the same id (e.g. a piecewise pattern where all predicates
// failed), there is no single "correct" row to surface. The first row is a
// reasonable default — it carries the primary gating reason. Policy authors who
// need different context in the report should add an explicit catch-all row.
static Classification make_fallback(const std::string& first_when,
                                    const std::string& first_ticket,
                                    const std::string& first_reason) {
  Classification c;
  c.cls    = is_software_predicate(first_when) ? Class::Skip : Class::Na;
  c.ticket = first_ticket;
  c.reason = first_reason.empty()
      ? "no predicate matched (environment does not satisfy any when clause)"
      : first_reason;
  return c;
}

Classification Policy::classify(const std::string& test_id,
                                const EnvironmentInfo& env,
                                bool* found) const {
  if (found) *found = false;

  if (!present) {
    // Discovery mode: treat as MUST_PASS (gate active). Caller sets found=true
    // so all tests are selected.
    if (found) *found = true;
    Classification c;
    c.cls = Class::MustPass;
    return c;
  }

  // 1. Exact id match: first row in declaration order whose `when` passes.
  const Entry* first_exact = nullptr;  // first row that matched by id (pred aside)
  for (const Entry& e : entries_) {
    if (is_glob(e.test) || e.test != test_id) continue;
    if (!first_exact) first_exact = &e;
    if (eval_when(e.when, env)) {
      if (found) *found = true;
      Classification result = e.cls;
      result.when_matched = e.when;
      return result;
    }
  }
  if (first_exact) {
    if (found) *found = true;
    return make_fallback(first_exact->when, first_exact->cls.ticket,
                         first_exact->cls.reason);
  }

  // 2. Wildcard glob: longest literal prefix; ties -> first-declared.
  //    Two-pass: Pass A finds the best prefix; Pass B picks the first passing row.
  std::size_t best_prefix = 0;
  const Entry* first_glob = nullptr;
  for (const Entry& e : entries_) {
    if (!is_glob(e.test)) continue;
    if (::fnmatch(e.test.c_str(), test_id.c_str(), 0) != 0) continue;
    const std::size_t plen = literal_prefix_len(e.test);
    if (!first_glob || plen > best_prefix) {
      best_prefix = plen;
      first_glob = &e;
    }
  }
  if (first_glob) {
    for (const Entry& e : entries_) {
      if (!is_glob(e.test)) continue;
      if (literal_prefix_len(e.test) != best_prefix) continue;
      if (::fnmatch(e.test.c_str(), test_id.c_str(), 0) != 0) continue;
      if (eval_when(e.when, env)) {
        if (found) *found = true;
        Classification result = e.cls;
        result.when_matched = e.when;
        return result;
      }
    }
    if (found) *found = true;
    return make_fallback(first_glob->when, first_glob->cls.ticket,
                         first_glob->cls.reason);
  }

  // 3. Truly unlisted -> not run (found stays false).
  return Classification{};
}

}  // namespace acceptance
