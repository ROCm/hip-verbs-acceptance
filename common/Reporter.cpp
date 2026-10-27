/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/Reporter.hpp"

#include "common/Version.hpp"
#include "common/version_info.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace acceptance {

Status reduce(const std::vector<Status>& per_rank) {
  bool any = false, any_fail = false, any_skip = false;
  for (Status s : per_rank) {
    if (s == Status::None) continue;  // identity: drop before reducing
    any = true;
    if (s == Status::Fail) any_fail = true;
    else if (s == Status::Skip) any_skip = true;
  }
  if (!any) return Status::None;  // all passive / no participants
  if (any_fail) return Status::Fail;
  if (any_skip) return Status::Skip;
  return Status::Pass;
}

Verdict apply_classification(Status reduced, const Classification& c) {
  // Discovery mode is handled in Policy::classify() — when no policy
  // file is present all tests get Class::MustPass. There is no gate-disable
  // path here; the parameter was removed to avoid misleading readers.
  Verdict v;

  switch (c.cls) {
    case Class::MustPass:
      switch (reduced) {
        case Status::Pass: v.status = FinalStatus::Pass; break;
        case Status::Fail: v.status = FinalStatus::Fail; v.gate_fail = true; break;
        // Unexpected skip of MUST_PASS is a gate failure.
        case Status::Skip:
          v.status = FinalStatus::Fail; v.gate_fail = true;
          v.note = "unexpected skip of a MUST_PASS test";
          break;
        // No rank ran -> na, not fail.
        case Status::None: v.status = FinalStatus::Na; break;
      }
      break;

    case Class::KnownFailure: {
      // KNOWN_FAILURE outputs as Fail to create fix pressure. The
      // detail field carries context (ticket, "known failure possibly fixed").
      // Gate always fails — expected or not.
      const std::string expected =
          c.expected_status.empty() ? "fail" : c.expected_status;
      const std::string actual = to_string(reduced);
      v.status = FinalStatus::Fail;
      v.gate_fail = (reduced != Status::None);  // none -> na, not fail
      if (reduced == Status::None) {
        v.status = FinalStatus::Na;
        v.gate_fail = false;
      } else if (actual != expected) {
        v.note = "known failure possibly fixed";
      } else {
        v.note = "known failure";
      }
      break;
    }

    case Class::Skip:
      // Software-gated; gate-neutral. Shown as [ SKIP ] with reason.
      v.status   = FinalStatus::Skip;
      v.gate_fail = false;
      break;

    case Class::Na:
      // Hardware-permanent; not run; synthesized as not-applicable.
      v.status = FinalStatus::Na;
      break;
  }
  return v;
}

// Hand-emitted JSON: escape strings; locale-independent numbers.
namespace {

// Escape a string for inclusion inside JSON double-quotes. Policy reasons are
// untrusted file content, so this also guarantees the output is valid UTF-8:
// well-formed multi-byte sequences pass through, malformed bytes become U+FFFD.
std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  const size_t n = s.size();
  for (size_t i = 0; i < n; ) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    switch (c) {
      case '"':  out += "\\\""; ++i; continue;
      case '\\': out += "\\\\"; ++i; continue;
      case '\n': out += "\\n";  ++i; continue;
      case '\t': out += "\\t";  ++i; continue;
      case '\r': out += "\\r";  ++i; continue;
      case '\b': out += "\\b";  ++i; continue;
      case '\f': out += "\\f";  ++i; continue;
      default: break;
    }
    if (c < 0x20 || c == 0x7f) {
      // Control characters (incl. DEL) -> \u00XX.
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
      ++i;
      continue;
    }
    if (c < 0x80) {  // printable ASCII
      out += static_cast<char>(c);
      ++i;
      continue;
    }
    // Multi-byte UTF-8: validate the whole sequence; pass it through raw if it is
    // well-formed (no overlong forms, surrogates, or out-of-range code points),
    // otherwise emit one U+FFFD and resync at the next byte.
    int len = 0;
    unsigned int min_cp = 0, mask = 0;
    if      ((c & 0xE0) == 0xC0) { len = 2; min_cp = 0x80;    mask = 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; min_cp = 0x800;   mask = 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; min_cp = 0x10000; mask = 0x07; }

    bool valid = len != 0 && (i + static_cast<size_t>(len) <= n);
    unsigned int cp = 0;
    if (valid) {
      cp = c & mask;
      for (int k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) { valid = false; break; }
        cp = (cp << 6) | (cc & 0x3F);
      }
    }
    if (valid && cp >= min_cp && cp <= 0x10FFFF &&
        !(cp >= 0xD800 && cp <= 0xDFFF)) {
      out.append(s, i, static_cast<size_t>(len));
      i += static_cast<size_t>(len);
    } else {
      out += "\xEF\xBF\xBD";  // U+FFFD REPLACEMENT CHARACTER
      ++i;
    }
  }
  return out;
}

// A JSON string value: "<escaped>".
std::string jstr(const std::string& s) {
  return "\"" + json_escape(s) + "\"";
}

// A JSON string-or-null: null for empty, else the escaped string.
std::string jstr_or_null(const std::string& s) {
  if (s.empty()) return "null";
  return jstr(s);
}

// Locale-independent number formatting via std::to_chars.
std::string jnum(double v) {
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  if (res.ec != std::errc{}) return "0";
  return std::string(buf, res.ptr);
}

std::string jnum(long long v) {
  char buf[32];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  if (res.ec != std::errc{}) return "0";
  return std::string(buf, res.ptr);
}

std::string jbool(bool b) { return b ? "true" : "false"; }

// A perf metric value: emit a bare JSON number when the string parses fully as a
// double (the median/min/... stats), a bare JSON array when it is a pre-formatted
// "[...]" list (the ".samples" raw runs), else a JSON string (".unit").
std::string jperf_val(const std::string& s) {
  if (s.empty()) return "null";
  if (s.size() >= 2 && s.front() == '[' && s.back() == ']') return s;  // array
  double d = 0.0;
  const char* first = s.data();
  const char* last = s.data() + s.size();
  auto res = std::from_chars(first, last, d);
  if (res.ec == std::errc{} && res.ptr == last) return s;  // numeric
  return jstr(s);
}

}  // namespace

// ---------------------------------------------------------------------------
// Shared per-test line formatter (public — see Reporter.hpp). Used by both the
// Reporter end-block and the Runner live stream so the lines are identical.
// ---------------------------------------------------------------------------

// GTest-style fixed-width status tag for the per-test console line.
const char* status_tag(FinalStatus s) {
  switch (s) {
    case FinalStatus::Pass: return " PASS  ";
    case FinalStatus::Fail: return " FAIL  ";
    case FinalStatus::Skip: return " SKIP  ";
    case FinalStatus::Na:   return "  NA   ";
  }
  return " ????  ";
}

std::string format_test_line(const std::string& id, FinalStatus status,
                             const std::string& detail) {
  std::string line = "[ ";
  line += status_tag(status);
  line += " ] ";
  line += id;
  if (!detail.empty()) { line += "  ("; line += detail; line += ")"; }
  return line;
}

std::string format_group_header(const std::string& category,
                                std::string& last_category) {
  if (category == last_category) return "";
  last_category = category;
  return "\n--- " + category + " ---";
}

std::string pick_detail(const Verdict& v, const std::string& synth,
                        const std::string& rank_reason,
                        const std::string& cls_reason) {
  if (!v.note.empty())        return v.note;        // e.g. "possibly fixed"
  if (!synth.empty())         return synth;         // crash|timeout|corrupt
  if (!rank_reason.empty())   return rank_reason;   // test's fail/skip reason
  if (v.status == FinalStatus::Pass) return std::string();  // policy reason is not a pass explanation
  if (!cls_reason.empty())    return cls_reason;    // policy reason (na/known)
  return std::string();
}

const EnvironmentInfo& pick_predicate_env(
    const std::vector<ReportInputs::NodeInfo>& nodes) {
  // Lowest-rank node's environment, chosen explicitly so the choice is
  // independent of insertion order. Returned by reference (no copy of the
  // potentially large rdma/gpu/affinity vectors + symbols map). When `nodes` is
  // empty, a static empty env (all fields empty -> only catch-all `when` rows
  // match). `homogeneous` means all nodes agree on the key dims, so the lowest
  // rank is a stable representative.
  static const EnvironmentInfo kEmptyEnv;
  const EnvironmentInfo* best = &kEmptyEnv;
  int best_rank = 0;
  for (const auto& n : nodes) {
    if (best == &kEmptyEnv || n.rank < best_rank) {
      best = &n.environment;
      best_rank = n.rank;
    }
  }
  return *best;
}

int write_report(const Policy& policy, const ReportInputs& in,
                 const std::string& report_path) {
  const bool discovery = in.discovery;

  // We assemble the tests[] array into a string buffer as we iterate, so the
  // gate verdict and the JSON share a single pass.
  std::ostringstream tests_json;
  bool first_test = true;
  std::vector<std::string> console_lines;  // one GTest-style line per test (+ headers)
  std::string last_console_category;
  std::vector<PerfTestRow> perf_rows;      // measured metrics, for the perf block

  int n_pass = 0, n_fail = 0, n_skip = 0, n_na = 0;
  bool gate_fail_any = false;
  int n_flagged = 0;

  // Env for `when` predicate evaluation — the lowest-rank node's,
  // chosen deterministically (see pick_predicate_env). Reference, no copy.
  const EnvironmentInfo& pred_env = pick_predicate_env(in.nodes);

  // Iterate the registry but emit a row only for SELECTED tests (the same set
  // the worker ran). Tests excluded by --filter/--kind are NOT in selected_ids
  // and are skipped entirely — they must not be synthesized into the report
  // (otherwise an excluded MUST_PASS failure would surface as `na` and silently
  // pass the gate). For a full unfiltered run selected_ids == the whole registry.
  const bool has_class_filter = !in.class_filter.empty();
  const Class filter_cls =
      has_class_filter ? class_from(in.class_filter) : Class::Na;
  for (const auto& meta : TestRegistry::all()) {
    if (in.selected_ids.find(meta.id) == in.selected_ids.end()) continue;

    bool in_policy = false;
    const Classification cls = policy.classify(meta.id, pred_env, &in_policy);
    // Unlisted tests are not run and do not appear in the report.
    if (!in_policy) continue;
    // --filter-class: the worker did not run tests of another class, so they
    // are not part of this report (not a missing result).
    if (has_class_filter && cls.cls != filter_cls) continue;

    // Reduced status + per-rank inputs.
    Status reduced;
    std::ostringstream reduced_from;  // JSON object body: "0":"pass","1":"fail"
    bool first_rank = true;
    std::string synth;
    std::string rank_reason;          // first non-empty per-rank reason (for console)
    double duration_ms = 0;
    // Every reporting rank's metrics, merged (lowest rank unprefixed, the rest
    // under "rank<N>."). Collected per rank first so merge_rank_perf sees them
    // in rank order.
    std::vector<std::pair<int, PerfSink>> per_rank_perf;
    PerfSink perf;
    bool no_result = false;           // selected + non-NA but no rank produced a row

    auto it = in.rank_results.find(meta.id);
    if (it != in.rank_results.end() && !it->second.empty()) {
      // Test was run by one or more ranks. Reduce over exactly the ranks that
      // produced a row (§7.3); reduced_from carries each contributing rank's
      // raw per-rank status (pass|fail|skip|none only). Iterating a std::map
      // yields ranks in ascending order, so reduced_from is {"0":..,"1":..}.
      std::vector<Status> per_rank;
      per_rank.reserve(it->second.size());
      for (const auto& [rk, r] : it->second) {
        per_rank.push_back(r.status);
        if (!first_rank) reduced_from << ",";
        first_rank = false;
        reduced_from << jstr(std::to_string(rk)) << ":"
                     << jstr(to_string(r.status));
        // Surface any synthesized failure marker, a reason, and the longest duration.
        if (!r.synth.empty()) synth = r.synth;
        if (rank_reason.empty() && !r.reason.empty()) rank_reason = r.reason;
        if (r.duration_ms > duration_ms) duration_ms = r.duration_ms;
        // Perf metrics: keep EVERY rank's. A 2-node test where both sides
        // measure (each initiating its own traffic) reports two sets of numbers
        // and dropping one loses half the run.
        if (!r.perf.empty()) per_rank_perf.emplace_back(rk, r.perf);
      }
      perf = merge_rank_perf(per_rank_perf);
      reduced = reduce(per_rank);
    } else if (cls.cls == Class::Na || cls.cls == Class::Skip) {
      // NA synthesis: not run, no rank participated.
      reduced = Status::None;
      // reduced_from intentionally empty: no rank produced a result.
    } else {
      // Selected, not NA, but NO rank produced a row. This is a STRUCTURAL gap
      // (truncated/unparsed worker output, a parse bug, or a worker that died
      // before emitting this test) — NOT the benign "all ranks passive" case,
      // which lands in the reduce branch above with real None rows. We must NOT
      // let it collapse to None -> na (which would silently pass the gate).
      reduced = Status::None;
      no_result = true;
    }

    Verdict v = apply_classification(reduced, cls);
    if (no_result) {
      // Override the None-based classification: a missing result is a structural
      v.status = FinalStatus::Fail;
      v.flagged = true;
      v.gate_fail = true;
      v.note = "no result produced by any rank (structural failure)";
      synth = "missing";
    }

    if (v.gate_fail) gate_fail_any = true;
    if (v.flagged) ++n_flagged;

    switch (v.status) {
      case FinalStatus::Pass: ++n_pass; break;
      case FinalStatus::Fail: ++n_fail; break;
      case FinalStatus::Skip: ++n_skip; break;
      case FinalStatus::Na:   ++n_na;   break;
    }

    const Class report_cls = cls.cls;

    // `reason`: prefer the test's RUNTIME fail/skip message (rank_reason) so a
    // real failure keeps its explanation in the JSON artifact; fall back to the
    // policy classification reason (e.g. an NA/known-failure rationale) when the
    // test produced none. The policy reason is suppressed on Pass (it explains a
    // non-run/known outcome, not a pass), so a passing row is "reason": null.
    // Mirrors pick_detail()'s console precedence and the documented schema.
    const std::string report_reason =
        !rank_reason.empty() ? rank_reason
        : (v.status == FinalStatus::Pass ? std::string() : cls.reason);

    // Reassembled metrics for the console block / CSV. Tests with no measured
    // metric (or only PerfProbe call timings, which parse_perf_metrics drops)
    // contribute no row.
    if (!perf.empty()) {
      PerfTestRow row{meta.category, meta.id, parse_perf_metrics(perf)};
      if (!row.metrics.empty()) perf_rows.push_back(std::move(row));
    }

    // "perf": { "<key>": <num|str>, ... } (empty {} for functional tests).
    std::ostringstream perf_json;
    {
      bool first_perf = true;
      for (const auto& kv : perf) {
        perf_json << (first_perf ? "" : ",") << "\n        " << jstr(kv.first)
                  << ": " << jperf_val(kv.second);
        first_perf = false;
      }
    }

    if (!first_test) tests_json << ",";
    first_test = false;
    tests_json
        << "\n    {"
        << "\n      \"id\": "              << jstr(meta.id) << ","
        << "\n      \"category\": "        << jstr(meta.category) << ","
        << "\n      \"class\": "           << jstr(to_string(report_cls)) << ","
        << "\n      \"kind\": "            << jstr(to_string(meta.kind)) << ","
        << "\n      \"topology\": "        << jstr(to_string(meta.topology)) << ","
        << "\n      \"status\": "          << jstr(to_string(v.status)) << ","
        << "\n      \"reduced_from\": {"   << reduced_from.str() << "},"
        << "\n      \"expected_status\": " << jstr_or_null(cls.expected_status) << ","
        << "\n      \"reason\": "          << jstr_or_null(report_reason) << ","
        << "\n      \"when_matched\": "    << jstr_or_null(cls.when_matched) << ","
        << "\n      \"ticket\": "          << jstr_or_null(cls.ticket) << ","
        << "\n      \"synth\": "           << jstr(synth) << ","
        << "\n      \"duration_ms\": "     << jnum(duration_ms) << ","
        << "\n      \"perf\": {"           << perf_json.str()
        << (perf.empty() ? "" : "\n      ") << "},"
        << "\n      \"gate_fail\": "       << jbool(v.gate_fail) << ","
        << "\n      \"flagged\": "         << jbool(v.flagged) << ","
        << "\n      \"note\": "            << jstr_or_null(v.note)
        << "\n    }";

    // GTest-style per-test console line: "[ TAG ] id  (detail)". Built via the
    // shared formatter so it is byte-identical to the Runner's live stream.
    const std::string detail = pick_detail(v, synth, rank_reason, cls.reason);
    const std::string hdr = format_group_header(meta.category, last_console_category);
    if (!hdr.empty()) console_lines.push_back(hdr);
    console_lines.push_back(format_test_line(meta.id, v.status, detail));
  }

  // nodes[]: the actual ranks/hosts that participated, sorted by rank. Falls back
  // to a single localhost/rank-0 entry for a bare single-node run that did not
  // populate nodes. Each node carries its DETECTED EnvironmentInfo (M1a/M1b).
  std::vector<ReportInputs::NodeInfo> nodes = in.nodes;
  if (nodes.empty()) nodes.push_back({0, "localhost", EnvironmentInfo{}});
  std::sort(nodes.begin(), nodes.end(),
            [](const ReportInputs::NodeInfo& a, const ReportInputs::NodeInfo& b) {
              return a.rank < b.rank;
            });
  std::ostringstream nodes_json;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const EnvironmentInfo& env = nodes[i].environment;
    std::ostringstream rdma_json;
    for (size_t j = 0; j < env.rdma_devices.size(); ++j) {
      const RdmaDevice& d = env.rdma_devices[j];
      rdma_json
          << (j ? "," : "")
          << "\n          {"
          << "\n            \"device\": "    << jstr_or_null(d.device) << ","
          << "\n            \"netdev\": "    << jstr_or_null(d.netdev) << ","
          << "\n            \"fw_ver\": "    << jstr_or_null(d.fw_ver) << ","
          << "\n            \"board_id\": "  << jstr_or_null(d.board_id) << ","
          << "\n            \"node_guid\": " << jstr_or_null(d.node_guid) << ","
          << "\n            \"gid_index\": "
          << jnum(static_cast<long long>(d.gid_index)) << ","
          << "\n            \"gid_type\": "  << jstr_or_null(d.gid_type) << ","
          << "\n            \"port_state\": " << jstr_or_null(d.port_state) << ","
          // vendor_id/vendor_part_id are -1 when not queried (default build):
          // emit null in that case, the numeric value otherwise.
          << "\n            \"vendor_id\": "
          << (d.vendor_id < 0 ? "null"
                              : jnum(static_cast<long long>(d.vendor_id)))
          << ","
          << "\n            \"vendor_part_id\": "
          << (d.vendor_part_id < 0
                  ? "null"
                  : jnum(static_cast<long long>(d.vendor_part_id)))
          << "\n          }";
    }

    // symbols: { "<name>": bool, ... } in map (sorted) order.
    std::ostringstream sym_json;
    {
      bool first_sym = true;
      for (const auto& [name, present] : env.symbols) {
        sym_json << (first_sym ? "" : ",") << "\n          " << jstr(name)
                 << ": " << jbool(present);
        first_sym = false;
      }
    }
    nodes_json
        << (i ? "," : "")
        << "\n    {"
        << "\n      \"host\": " << jstr(nodes[i].host) << ","
        << "\n      \"rank\": " << jnum(static_cast<long long>(nodes[i].rank)) << ","
        << "\n      \"environment\": {"
        << "\n        \"vendor\": "       << jstr_or_null(env.vendor) << ","
        << "\n        \"rdma_prefix\": "  << jstr_or_null(env.rdma_prefix) << ","
        << "\n        \"rocm_version\": " << jstr_or_null(env.rocm_version) << ","
        << "\n        \"rocm_path\": "    << jstr_or_null(env.rocm_path) << ","
        << "\n        \"rocm_version_source\": " << jstr_or_null(env.rocm_version_source) << ","
        << "\n        \"gpu_count\": "    << jnum(static_cast<long long>(env.gpu_count)) << ","
        << "\n        \"kernel\": "       << jstr_or_null(env.kernel) << ","
        << "\n        \"os\": "           << jstr_or_null(env.os) << ","
        << "\n        \"gpu_model\": "    << jstr_or_null(env.gpu_model) << ","
        << "\n        \"driver_version\": " << jstr_or_null(env.driver_version) << ","
        << "\n        \"symbols\": {" << sym_json.str()
        << (env.symbols.empty() ? "" : "\n        ") << "},"
        << "\n        \"rdma_devices\": [" << rdma_json.str()
        << (env.rdma_devices.empty() ? "" : "\n        ") << "],"
        << "\n        \"gpu_devices\": [";
    for (std::size_t j = 0; j < env.gpu_devices.size(); ++j) {
      const GpuDevice& g = env.gpu_devices[j];
      nodes_json
          << (j ? "," : "")
          << "\n          {"
          << "\n            \"ordinal\": "   << jnum(static_cast<long long>(g.ordinal)) << ","
          << "\n            \"hip_ordinal\": " << jnum(static_cast<long long>(g.hip_ordinal)) << ","
          << "\n            \"bdf\": "       << jstr_or_null(g.bdf) << ","
          << "\n            \"numa_node\": " << jnum(static_cast<long long>(g.numa_node)) << ","
          << "\n            \"arch\": "      << jstr_or_null(g.arch)
          << "\n          }";
    }
    nodes_json
        << (env.gpu_devices.empty() ? "" : "\n        ") << "],"
        << "\n        \"gpu_nic_affinity\": [";
    for (std::size_t j = 0; j < env.gpu_nic_affinity.size(); ++j) {
      const GpuNicAffinity& a = env.gpu_nic_affinity[j];
      nodes_json
          << (j ? "," : "")
          << " {\"gpu\": " << jnum(static_cast<long long>(a.gpu_ordinal))
          << ", \"nearest_nic\": " << jstr_or_null(a.nearest_nic)
          << ", \"ambiguous\": " << jbool(a.ambiguous) << "}";
    }
    const topology::Pair& pair = env.rank_pair;
    nodes_json
        << "]"
        << "\n      },"
        << "\n      \"topology\": {"
        << "\n        \"hca\": "        << jstr_or_null(pair.hca) << ","
        << "\n        \"gpu\": "        << jnum(static_cast<long long>(pair.gpu)) << ","
        << "\n        \"hca_source\": " << jstr_or_null(pair.hca_source) << ","
        << "\n        \"gpu_source\": " << jstr_or_null(pair.gpu_source) << ","
        << "\n        \"local\": "      << jbool(pair.local) << ","
        << "\n        \"note\": "       << jstr_or_null(pair.note)
        << "\n      }"
        << "\n    }";
  }

  // homogeneous: true iff every node agrees on the KEY environment dims
  // {vendor, rdma_prefix, rocm_version, gpu_count, gpu_model, driver_version, the
  // SET of device fw_ver, and the symbols map (probed feature set)}.
  // Per-node-unique fields (node_guid, netdev, host, rank) are deliberately NOT
  // compared. A single node is trivially homogeneous.
  // fw_ver is compared as an unordered SET, not per-device: two nodes are
  // considered fw-homogeneous if they expose the same set of firmware versions,
  // regardless of which device carries which (the intent is "one fw level across
  // the cluster", not "device N matches device N").
  auto fw_set = [](const EnvironmentInfo& e) {
    std::set<std::string> s;
    for (const RdmaDevice& d : e.rdma_devices) s.insert(d.fw_ver);
    return s;
  };
  bool homogeneous = true;
  for (size_t i = 1; i < nodes.size(); ++i) {
    const EnvironmentInfo& a = nodes[0].environment;
    const EnvironmentInfo& b = nodes[i].environment;
    if (a.vendor != b.vendor || a.rdma_prefix != b.rdma_prefix ||
        a.rocm_version != b.rocm_version || a.gpu_count != b.gpu_count ||
        a.gpu_model != b.gpu_model || a.driver_version != b.driver_version ||
        a.symbols != b.symbols || fw_set(a) != fw_set(b)) {
      homogeneous = false;
      break;
    }
  }

  std::string launch_hosts = "[";
  for (size_t i = 0; i < in.launch.hosts.size(); ++i)
    launch_hosts += (i ? ", " : "") + jstr(in.launch.hosts[i]);
  launch_hosts += "]";

  // Assemble the full document.
  std::ostringstream doc;
  doc << "{"
      << "\n  \"schema_version\": " << jstr("1.3") << ","
      << "\n  \"harness\": {"
      << "\n    \"version\": " << jstr(kHarnessVersion) << ","
      << "\n    \"git_sha\": "  << jstr(kGitSha)
      << "\n  },"
      << "\n  \"mode\": " << jstr(discovery ? "discovery" : "gate") << ","
      << "\n  \"filtered\": " << jbool(in.filtered) << ","
      << "\n  \"launch\": {"
      << "\n    \"np\": "    << jnum(static_cast<long long>(in.launch.np)) << ","
      << "\n    \"ppn\": "   << jnum(static_cast<long long>(in.launch.ppn)) << ","
      << "\n    \"hosts\": " << launch_hosts
      << "\n  },"
      << "\n  \"summary\": {"
      << "\n    \"passed\": "         << jnum(static_cast<long long>(n_pass)) << ","
      << "\n    \"failed\": "         << jnum(static_cast<long long>(n_fail)) << ","
      << "\n    \"skipped\": "        << jnum(static_cast<long long>(n_skip)) << ","
      << "\n    \"not_applicable\": " << jnum(static_cast<long long>(n_na))
      << "\n  },"
      << "\n  \"homogeneous\": " << jbool(homogeneous) << ","
      << "\n  \"nodes\": [" << nodes_json.str()
      << "\n  ],"
      << "\n  \"tests\": [" << tests_json.str()
      << (first_test ? "" : "\n  ") << "]"
      << "\n}\n";

  // Write report.json.
  {
    std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::fprintf(stderr, "error: cannot open report file '%s'\n",
                   report_path.c_str());
      return 1;
    }
    out << doc.str();
    out.flush();
    if (!out) {
      std::fprintf(stderr, "error: failed writing report file '%s'\n",
                   report_path.c_str());
      return 1;
    }
  }

  // Per-test console lines (GTest-style), unless --quiet OR they were already
  // streamed live by the Runner (single-node path). The summary below always
  // prints.
  if (!in.quiet && !in.per_test_already_streamed) {
    for (const std::string& l : console_lines) std::printf("%s\n", l.c_str());
  }

  // Performance block. Unlike the per-test lines this is ALWAYS printed here
  // (the Runner's live stream carries no metrics — they are only complete once
  // every rank's result has been reconciled), so a --kind perf run finally shows
  // its numbers without post-processing report.json.
  if (!in.quiet) {
    for (const std::string& l : render_perf_section(perf_rows))
      std::printf("%s\n", l.c_str());
  }

  // Perf CSV, when asked for. Written even when the run measured nothing (a
  // header-only file) so a caller scripting around it always finds the file.
  // A write failure is loud (write_perf_csv prints it) but does NOT change the
  // exit code: that code means "the acceptance gate passed", and perf is
  // report-only.
  if (!in.perf_csv.empty() && write_perf_csv(in.perf_csv, perf_rows))
    std::printf("perf csv: %s\n", in.perf_csv.c_str());

  // Human summary.
  const bool gate_pass = !gate_fail_any;
  std::printf(
      "report: mode=%s | pass=%d fail=%d skip=%d na=%d | flagged=%d | -> %s\n",
      in.discovery ? "discovery" : "gate",
      n_pass, n_fail, n_skip, n_na, n_flagged, report_path.c_str());
  if (gate_pass) {
    std::printf("ACCEPTANCE: PASS\n");
  } else {
    std::printf("ACCEPTANCE: FAIL — %d test(s) failed the gate\n", n_fail);
  }
  return gate_pass ? 0 : 1;
}

}  // namespace acceptance
