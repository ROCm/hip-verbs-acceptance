/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_resultio — round-trip tests for TestResult key=value (de)serialization and
// the per-rank result_io::serialize/parse (rank header + @test blocks). Covers
// reasons containing '=' and ',', empty-reason and last-record handling, the
// missing-rank-header => parse-false rule, and the one_line() newline guard that
// stops a multi-line value from injecting a phantom record.

#include <sstream>
#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/ResultIO.hpp"
#include "common/TestResult.hpp"
#include "common/Types.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

namespace accept_test {

namespace {

// Serialize one TestResult to its key=value block, parse it back.
TestResult kv_round_trip(const TestResult& in) {
  std::ostringstream os;
  in.to_kv(os);
  std::istringstream is(os.str());
  return TestResult::from_kv(is);
}

void check_result_eq(const TestResult& got, const TestResult& want,
                     const char* ctx) {
  CHECK_EQ_STR(got.id, want.id, (std::string(ctx) + " id").c_str());
  CHECK_EQ_STR(to_string(got.status), to_string(want.status),
               (std::string(ctx) + " status").c_str());
  CHECK_EQ_STR(got.reason, want.reason, (std::string(ctx) + " reason").c_str());
  CHECK(got.duration_ms == want.duration_ms,
        (std::string(ctx) + " duration_ms").c_str());
  CHECK(got.exit_code == want.exit_code,
        (std::string(ctx) + " exit_code").c_str());
  CHECK_EQ_STR(got.synth, want.synth, (std::string(ctx) + " synth").c_str());
}

void test_testresult_round_trip() {
  // Basic record with all fields populated.
  TestResult a;
  a.id = "syn_x";
  a.status = Status::Fail;
  a.reason = "boom";
  a.duration_ms = 12.5;
  a.exit_code = 7;
  a.synth = "crash";
  check_result_eq(kv_round_trip(a), a, "basic");

  // Reason containing both '=' and ',' (the split-on-first-'=' rule must keep
  // everything after the first '=' as the value, commas and all).
  TestResult b;
  b.id = "syn_eqcomma";
  b.status = Status::Skip;
  b.reason = "k=v, also a=b, x>y";
  b.duration_ms = 0;
  b.exit_code = 0;
  b.synth = "";
  check_result_eq(kv_round_trip(b), b, "reason with = and ,");

  // Empty reason / empty synth survive as empty.
  TestResult c;
  c.id = "syn_empty";
  c.status = Status::Pass;
  c.reason = "";
  c.synth = "";
  check_result_eq(kv_round_trip(c), c, "empty reason/synth");
}

void test_one_line_guard() {
  // A reason with an embedded newline must be sanitized to ONE line so it cannot
  // inject a phantom record / extra key=value line on parse.
  TestResult a;
  a.id = "syn_nl";
  a.status = Status::Fail;
  a.reason = "line1\nstatus=pass\nline3";  // a forged status= line in the reason
  a.synth = "with\r\nreturn";
  std::ostringstream os;
  a.to_kv(os);
  const std::string text = os.str();
  // The serialized block must contain exactly the contractual key lines (id,
  // status, reason, duration_ms, exit_code, synth) — newlines in values were
  // replaced by spaces, so no extra line was introduced.
  int line_count = 0;
  {
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) ++line_count;
  }
  CHECK(line_count == 6, "one_line guard: exactly 6 key lines (no injection)");

  // Round-trip: status must remain Fail (NOT overwritten by the forged
  // 'status=pass' that was embedded in the reason), proving no phantom field.
  std::istringstream is(text);
  const TestResult got = TestResult::from_kv(is);
  CHECK(got.status == Status::Fail,
        "one_line guard: embedded 'status=pass' did not override status");
  CHECK(got.reason.find('\n') == std::string::npos,
        "one_line guard: reason has no newline after sanitization");
}

void test_result_io_multi_record() {
  std::vector<TestResult> recs;
  TestResult r0;
  r0.id = "syn_a";
  r0.status = Status::Pass;
  r0.reason = "";  // empty-reason record
  r0.duration_ms = 1;
  recs.push_back(r0);

  TestResult r1;
  r1.id = "syn_b";
  r1.status = Status::Fail;
  r1.reason = "why, with comma";
  r1.duration_ms = 2.25;
  r1.exit_code = 3;
  r1.synth = "timeout";
  recs.push_back(r1);

  TestResult r2;  // last record must survive (no trailing separator)
  r2.id = "syn_c";
  r2.status = Status::Skip;
  r2.reason = "needs 2 ranks";
  recs.push_back(r2);

  const std::string text = result_io::serialize(5, recs);

  int rank = -1;
  std::vector<TestResult> got;
  const bool ok = result_io::parse(text, rank, got);
  CHECK(ok, "result_io::parse returns true with a rank header");
  CHECK(rank == 5, "result_io::parse recovers the rank header value");
  CHECK(got.size() == 3, "result_io::parse recovers all 3 records");
  if (got.size() == 3) {
    check_result_eq(got[0], r0, "multi rec0 (empty reason)");
    check_result_eq(got[1], r1, "multi rec1");
    check_result_eq(got[2], r2, "multi rec2 (last)");
  }
}

void test_result_io_no_rank_header() {
  // A block with @test records but NO `rank=` header is a structural error: the
  // launcher relies on parse() returning false to treat it as a run failure.
  TestResult r;
  r.id = "syn_a";
  r.status = Status::Pass;
  std::ostringstream os;
  os << "@test\n";
  r.to_kv(os);

  int rank = -1;
  std::vector<TestResult> got;
  const bool ok = result_io::parse(os.str(), rank, got);
  CHECK(!ok, "result_io::parse returns false when rank= header is missing");
}

void test_result_io_malformed_rank() {
  // A non-integer (or trailing-junk) rank= value is structurally invalid: parse
  // must FAIL rather than silently coercing to rank 0 (which would misattribute
  // results to the wrong rank and hide truncated/garbled output).
  std::vector<TestResult> got;
  for (const char* bad : {"rank=abc\n@test\nid=x\n", "rank=\n@test\nid=x\n",
                          "rank=3x\n@test\nid=x\n", "rank=1 2\n@test\nid=x\n"}) {
    int rank = -1;
    CHECK(!result_io::parse(bad, rank, got),
          "result_io::parse rejects a non-integer rank header");
  }
  // A clean integer still parses.
  int rank = -1;
  CHECK(result_io::parse("rank=2\n@test\nid=x\n", rank, got) && rank == 2,
        "result_io::parse accepts a valid integer rank header");
}

// An @env block round-trips through serialize(rank, results, env) -> parse, and
// the legacy 2-arg parse still recovers rank + results (back-compat). Exercises
// >=2 rdma_devices so the indexed flattening is covered.
void test_result_io_env_round_trip() {
  std::vector<TestResult> recs;
  TestResult r0;
  r0.id = "syn_a";
  r0.status = Status::Pass;
  recs.push_back(r0);
  TestResult r1;
  r1.id = "syn_b";
  r1.status = Status::Fail;
  r1.reason = "boom";
  recs.push_back(r1);

  EnvironmentInfo env;
  env.vendor = "thor";
  env.rdma_prefix = "bnxt_re";
  env.rocm_version = "6.4.0-fake";
  env.rocm_path = "/opt/rocm-6.4.0";
  env.rocm_version_source = "runtime_info";
  env.gpu_count = 8;
  env.kernel = "6.6.0";
  env.os = "Ubuntu 22.04";
  env.gpu_model = "gfx942";
  env.driver_version = "driver=60342134 runtime=60342134";
  // M1c symbol probes: mixed true/false to exercise the sym.<name> round-trip.
  env.symbols["libionic"] = false;
  env.symbols["hsa_amd_portable_export_dmabuf"] = true;
  env.symbols["hipMemGetHandleForAddressRange"] = true;
  RdmaDevice d0;
  d0.device = "bnxt_re0";
  d0.netdev = "eth0";
  d0.fw_ver = "230.0.1";
  d0.board_id = "BCM57608";
  d0.node_guid = "aaaa:bbbb:cccc:dddd";
  d0.gid_index = 3;
  d0.gid_type = "RoCE v2";
  d0.vendor_id = 0x14e4;       // Broadcom
  d0.vendor_part_id = 0x1750;  // exercises the ibverbs-enrichment fields
  env.rdma_devices.push_back(d0);
  RdmaDevice d1;
  d1.device = "bnxt_re1";
  d1.netdev = "eth1";
  d1.fw_ver = "230.0.1";
  d1.gid_index = 5;
  d1.gid_type = "RoCE v2";
  env.rdma_devices.push_back(d1);
  // >=2 GPU devices + affinity entries so the gpu./affinity. indexed flattening
  // is covered (these arrays were previously serialized but never round-tripped).
  GpuDevice g0;
  g0.ordinal = 0;
  g0.bdf = "0000:69:00.0";
  g0.numa_node = 0;
  g0.arch = "gfx942";
  env.gpu_devices.push_back(g0);
  GpuDevice g1;
  g1.ordinal = 1;
  g1.bdf = "0001:56:00.0";
  g1.numa_node = -1;  // exercises the -1 default round-trip
  g1.arch = "gfx942";
  env.gpu_devices.push_back(g1);
  env.gpu_nic_affinity.push_back({0, "bnxt_re0"});
  env.gpu_nic_affinity.push_back({1, "bnxt_re1"});

  const std::string text = result_io::serialize(7, recs, env);

  // 4-arg parse recovers env.
  int rank = -1;
  std::vector<TestResult> got;
  EnvironmentInfo got_env;
  const bool ok = result_io::parse(text, rank, got, got_env);
  CHECK(ok, "env round-trip: parse returns true");
  CHECK(rank == 7, "env round-trip: rank recovered");
  CHECK(got.size() == 2, "env round-trip: both test records recovered");
  if (got.size() == 2) {
    check_result_eq(got[0], r0, "env round-trip rec0");
    check_result_eq(got[1], r1, "env round-trip rec1");
  }
  CHECK_EQ_STR(got_env.vendor, "thor", "env round-trip: vendor");
  CHECK_EQ_STR(got_env.rocm_version, "6.4.0-fake", "env round-trip: rocm_version");
  CHECK_EQ_STR(got_env.rocm_path, "/opt/rocm-6.4.0", "env round-trip: rocm_path");
  CHECK_EQ_STR(got_env.rocm_version_source, "runtime_info",
               "env round-trip: rocm_version_source");
  CHECK(got_env.gpu_count == 8, "env round-trip: gpu_count");
  // M1c scalars round-trip.
  CHECK_EQ_STR(got_env.gpu_model, "gfx942", "env round-trip: gpu_model");
  CHECK_EQ_STR(got_env.driver_version, "driver=60342134 runtime=60342134",
               "env round-trip: driver_version (value with '=' and space)");
  // M1c symbols map round-trips with values preserved per name.
  CHECK(got_env.symbols.size() == 3, "env round-trip: 3 symbols recovered");
  CHECK(got_env.symbols.count("libionic") &&
            got_env.symbols.at("libionic") == false,
        "env round-trip: symbols[libionic]==false");
  CHECK(got_env.symbols.count("hsa_amd_portable_export_dmabuf") &&
            got_env.symbols.at("hsa_amd_portable_export_dmabuf") == true,
        "env round-trip: symbols[hsa_amd_portable_export_dmabuf]==true");
  CHECK(got_env.symbols.count("hipMemGetHandleForAddressRange") &&
            got_env.symbols.at("hipMemGetHandleForAddressRange") == true,
        "env round-trip: symbols[hipMemGetHandleForAddressRange]==true");
  CHECK(got_env.rdma_devices.size() == 2,
        "env round-trip: 2 rdma devices recovered");
  if (got_env.rdma_devices.size() == 2) {
    CHECK_EQ_STR(got_env.rdma_devices[0].device, "bnxt_re0",
                 "env round-trip: device0 name");
    CHECK(got_env.rdma_devices[0].gid_index == 3,
          "env round-trip: device0 gid_index");
    CHECK_EQ_STR(got_env.rdma_devices[0].gid_type, "RoCE v2",
                 "env round-trip: device0 gid_type");
    CHECK(got_env.rdma_devices[0].vendor_id == 0x14e4,
          "env round-trip: device0 vendor_id");
    CHECK(got_env.rdma_devices[0].vendor_part_id == 0x1750,
          "env round-trip: device0 vendor_part_id");
    CHECK_EQ_STR(got_env.rdma_devices[1].device, "bnxt_re1",
                 "env round-trip: device1 name");
    CHECK(got_env.rdma_devices[1].gid_index == 5,
          "env round-trip: device1 gid_index");
    // device1 left vendor_id/part_id unset -> default -1 survives round-trip.
    CHECK(got_env.rdma_devices[1].vendor_id == -1,
          "env round-trip: device1 vendor_id default -1");
  }
  // gpu_devices + gpu_nic_affinity round-trip.
  CHECK(got_env.gpu_devices.size() == 2,
        "env round-trip: 2 gpu devices recovered");
  if (got_env.gpu_devices.size() == 2) {
    CHECK_EQ_STR(got_env.gpu_devices[0].bdf, "0000:69:00.0",
                 "env round-trip: gpu0 bdf");
    CHECK(got_env.gpu_devices[0].numa_node == 0, "env round-trip: gpu0 numa");
    CHECK_EQ_STR(got_env.gpu_devices[1].arch, "gfx942",
                 "env round-trip: gpu1 arch");
    CHECK(got_env.gpu_devices[1].numa_node == -1,
          "env round-trip: gpu1 numa default -1");
  }
  CHECK(got_env.gpu_nic_affinity.size() == 2,
        "env round-trip: 2 affinity entries recovered");
  if (got_env.gpu_nic_affinity.size() == 2) {
    CHECK(got_env.gpu_nic_affinity[1].gpu_ordinal == 1,
          "env round-trip: affinity1 gpu_ordinal");
    CHECK_EQ_STR(got_env.gpu_nic_affinity[1].nearest_nic, "bnxt_re1",
                 "env round-trip: affinity1 nearest_nic");
  }

  // Back-compat: the 2-arg parse still recovers rank + results from an @env
  // stream (it simply ignores the env block).
  int rank2 = -1;
  std::vector<TestResult> got2;
  const bool ok2 = result_io::parse(text, rank2, got2);
  CHECK(ok2 && rank2 == 7 && got2.size() == 2,
        "env round-trip: 2-arg parse ignores @env, recovers rank+results");

  // Back-compat the other way: a stream with NO @env block yields an empty env.
  const std::string legacy = result_io::serialize(2, recs);
  int rank3 = -1;
  std::vector<TestResult> got3;
  EnvironmentInfo env3;
  const bool ok3 = result_io::parse(legacy, rank3, got3, env3);
  CHECK(ok3 && rank3 == 2 && got3.size() == 2,
        "no-env stream: 4-arg parse recovers rank+results");
  CHECK(env3.vendor.empty() && env3.rdma_devices.empty(),
        "no-env stream: empty EnvironmentInfo");
}

// A hostile/corrupt @env stream must not let a large indexed key (gpu.<i> /
// affinity.<i> / rdma.<i>) force a huge allocation. The @env block crosses the
// ssh trust boundary (a remote worker's stdout), so from_kv must bound memory by
// the number of lines present, not by the magnitude of the index. Regression for
// the unbounded vector-grow bug (fixed by reassembling via a sparse map).
void test_env_hostile_index_bounded() {
  std::vector<std::string> lines = {
      "vendor=thor",
      "gpu.1000000000.arch=x",        // ~1e9 — would OOM if vector-resized
      "affinity.999999999.nearest_nic=y",
      "rdma.2000000000.device=z",
  };
  const EnvironmentInfo env = EnvironmentInfo::from_kv(lines);
  // Each array holds exactly the one (sparse) entry that was present; no resize
  // to the index magnitude.
  CHECK(env.gpu_devices.size() == 1,
        "hostile index: gpu_devices bounded to entries present");
  CHECK(env.gpu_nic_affinity.size() == 1,
        "hostile index: gpu_nic_affinity bounded to entries present");
  CHECK(env.rdma_devices.size() == 1,
        "hostile index: rdma_devices bounded to entries present");
  CHECK_EQ_STR(env.vendor, "thor", "hostile index: scalar fields still parse");
}

}  // namespace

void run_resultio_tests() {
  test_testresult_round_trip();
  test_one_line_guard();
  test_result_io_multi_record();
  test_result_io_no_rank_header();
  test_result_io_malformed_rank();
  test_result_io_env_round_trip();
  test_env_hostile_index_bounded();
}

}  // namespace accept_test
