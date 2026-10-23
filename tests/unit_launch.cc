/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_launch — unit tests for rank placement and the worker argv
// (common/Launch.*): host_is_local, validate_launch, plan_ranks and
// build_worker_args (each forwarded flag appears once, for local and remote
// ranks alike).

#include <algorithm>
#include <string>
#include <vector>

#include "common/Launch.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

namespace accept_test {

namespace {

int count_of(const std::vector<std::string>& argv, const std::string& s) {
  return static_cast<int>(std::count(argv.begin(), argv.end(), s));
}

// Value following the first `flag`, or "<absent>".
std::string value_of(const std::vector<std::string>& argv,
                     const std::string& flag) {
  for (size_t i = 0; i + 1 < argv.size(); ++i)
    if (argv[i] == flag) return argv[i + 1];
  return "<absent>";
}

void test_host_is_local() {
  CHECK(host_is_local("localhost", "n0"), "localhost is local");
  CHECK(host_is_local("127.0.0.1", "n0"), "127.0.0.1 is local");
  CHECK(host_is_local("n0", "n0"), "exact hostname is local");
  CHECK(host_is_local("n0", "n0.cluster.example"),
        "short name matches a fully qualified self");
  CHECK(host_is_local("n0.cluster.example", "n0"),
        "fully qualified name matches a short self");
  CHECK(!host_is_local("n0.a.example", "n0.b.example"),
        "two different FQDNs sharing a first label are not the same host");
  CHECK(!host_is_local("n1", "n0"), "another host is remote");
  CHECK(!host_is_local("n01", "n0"), "a prefix of the name is not a match");
  CHECK(!host_is_local("n0", ""), "unknown self hostname: only localhost matches");
}

void test_validate_launch() {
  using V = std::vector<std::string>;
  CHECK(validate_launch(1, false, 0, {}).empty(), "default: 1 local rank");
  CHECK(validate_launch(2, true, 0, {}).empty(), "-n 2 local");
  CHECK(validate_launch(2, true, 1, V{"a", "b"}).empty(), "-n 2 -ppn 1 -hosts a,b");
  CHECK(validate_launch(2, true, 2, V{"b"}).empty(), "-n 2 -ppn 2 -hosts b");
  CHECK(validate_launch(2, true, 0, V{"a", "b"}).empty(),
        "-n 2 -hosts a,b (ppn defaults to 1)");
  CHECK(validate_launch(1, false, 0, V{"b"}).empty(), "-hosts b: one remote rank");

  CHECK(!validate_launch(3, true, 0, {}).empty(), "-n 3 rejected");
  CHECK(!validate_launch(0, true, 0, {}).empty(), "-n 0 rejected");
  CHECK(!validate_launch(2, true, 1, {}).empty(),
        "-n 2 -ppn 1 on one host rejected (does not fit)");
  CHECK(!validate_launch(2, true, 1, V{"a"}).empty(),
        "-n 2 -ppn 1 -hosts a rejected (does not fit)");
  CHECK(!validate_launch(2, true, 1, V{"a", "a"}).empty(),
        "duplicate host rejected");
  CHECK(!validate_launch(1, false, 0, V{"a", "b"}).empty(),
        "-hosts with 2 hosts but no -n rejected");
  CHECK(validate_launch(1, true, 0, V{"a", "b"}).empty(),
        "explicit -n 1 with 2 hosts is allowed");
}

void test_plan_ranks() {
  // Default: both ranks on localhost, local ranks 0 and 1.
  {
    const auto p = plan_ranks(2, 0, {}, "n0");
    CHECK(p.size() == 2, "-n 2: two placements");
    CHECK(p[0].host == "localhost" && p[1].host == "localhost",
          "-n 2: both on localhost");
    CHECK(p[0].local_rank == 0 && p[1].local_rank == 1,
          "-n 2: local ranks 0 and 1");
    CHECK(p[0].is_local && p[1].is_local, "-n 2: both local");
  }
  // One rank per host: block order, local rank 0 on each.
  {
    const auto p = plan_ranks(2, 1, {"n0", "n1"}, "n0.cluster");
    CHECK(p[0].rank == 0 && p[0].host == "n0" && p[0].local_rank == 0,
          "-ppn 1: rank 0 on n0, local rank 0");
    CHECK(p[1].rank == 1 && p[1].host == "n1" && p[1].local_rank == 0,
          "-ppn 1: rank 1 on n1, local rank 0");
    CHECK(p[0].is_local && !p[1].is_local, "-ppn 1: n0 local, n1 remote");
  }
  // Both ranks on one remote host.
  {
    const auto p = plan_ranks(2, 2, {"n1"}, "n0");
    CHECK(p[0].host == "n1" && p[1].host == "n1", "-ppn 2 -hosts n1: both on n1");
    CHECK(!p[0].is_local && !p[1].is_local, "-ppn 2 -hosts n1: both remote");
    CHECK(p[1].local_rank == 1, "-ppn 2 -hosts n1: rank 1 is local rank 1");
  }
  // ppn defaults to ceil(N / #hosts): -n 2 over 2 hosts -> 1 per host.
  {
    const auto p = plan_ranks(2, 0, {"n0", "n1"}, "n0");
    CHECK(p[1].host == "n1", "default ppn spreads -n 2 over 2 hosts");
  }
  // -ppn larger than needed packs everything on the first host.
  {
    const auto p = plan_ranks(2, 4, {"n0", "n1"}, "n0");
    CHECK(p[0].host == "n0" && p[1].host == "n0",
          "-ppn 4: both ranks on the first host (block order)");
  }
}

void test_build_worker_args() {
  WorkerParams wp;
  wp.policy = "policy.csv";  // RELATIVE -> must be made absolute
  wp.filter = "syn2_*";
  wp.filter_class = "must_pass";
  wp.filter_category = "hip/*";
  wp.hide_na = true;
  wp.kind = "func";
  wp.isolate = "all";
  wp.isolate_explicit = true;
  wp.gid = "3";
  wp.device = "bnxt_re0";
  wp.deadline_sec = 130;

  RankPlacement pl;
  pl.rank = 1;
  pl.local_rank = 1;
  pl.host = "localhost";
  pl.is_local = true;

  const auto a = build_worker_args(wp, pl, 2, "127.0.0.1:4000", "synthetic");
  CHECK(!a.empty() && a[0] == "--worker", "worker args start with --worker");
  CHECK(value_of(a, "--rank") == "1", "--rank is the placement's rank");
  CHECK(value_of(a, "--size") == "2", "--size forwarded");
  CHECK(value_of(a, "--rendezvous") == "127.0.0.1:4000", "--rendezvous forwarded");
  CHECK(value_of(a, "--results-out") == "-", "results always go to stdout");
  CHECK(value_of(a, "--vendor") == "synthetic", "--vendor forwarded");
  CHECK(value_of(a, "--policy").rfind('/', 0) == 0 &&
            value_of(a, "--policy").size() > std::string("/policy.csv").size(),
        "--policy made absolute");
  CHECK(value_of(a, "--filter") == "syn2_*", "--filter forwarded verbatim");
  CHECK(value_of(a, "--filter-class") == "must_pass", "--filter-class forwarded");
  CHECK(value_of(a, "--filter-category") == "hip/*", "--filter-category forwarded");
  CHECK(value_of(a, "--kind") == "func", "--kind forwarded");
  CHECK(value_of(a, "--isolate") == "all", "--isolate forwarded when explicit");
  CHECK(value_of(a, "--gid") == "3", "--gid forwarded");
  CHECK(value_of(a, "--device") == "bnxt_re0", "--device forwarded");
  CHECK(value_of(a, "--deadline-sec") == "130", "--deadline-sec forwarded");
  // Every flag exactly once: a second builder that also appended one (the
  // old --local-ranks / --nodes split) would show up here.
  for (const char* f : {"--worker", "--rank", "--size", "--rendezvous",
                        "--results-out", "--vendor", "--policy", "--filter",
                        "--filter-class", "--filter-category", "--hide-na",
                        "--kind", "--isolate", "--gid", "--device",
                        "--deadline-sec"}) {
    CHECK(count_of(a, f) == 1, (std::string("worker args hold ") + f + " once").c_str());
  }

  // Local and remote ranks get the same argv.
  RankPlacement remote = pl;
  remote.host = "n1";
  remote.is_local = false;
  CHECK(build_worker_args(wp, remote, 2, "127.0.0.1:4000", "synthetic") == a,
        "remote and local ranks get identical worker args");

  // Unset options are omitted entirely.
  {
    WorkerParams q;
    q.isolate = "all";
    const auto b = build_worker_args(q, pl, 1, "h:1", "synthetic");
    for (const char* f : {"--policy", "--filter", "--filter-class",
                          "--filter-category", "--hide-na", "--kind",
                          "--isolate", "--gid", "--device", "--deadline-sec"}) {
      CHECK(count_of(b, f) == 0, (std::string("unset option omitted: ") + f).c_str());
    }
  }
}

}  // namespace

void run_launch_tests() {
  test_host_is_local();
  test_validate_launch();
  test_plan_ranks();
  test_build_worker_args();
}

}  // namespace accept_test
