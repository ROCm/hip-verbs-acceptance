/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_topology — unit tests for common/Topology.*: the KFD BDF and PCIe
// distance helpers, select_pair over hand-built environments (affinity, ties,
// no affinity, non-visible GPUs, HIP vs KFD ordinals, an inactive nearest
// NIC), format_pair, validate_lists, get_rank_entry and the ACCEPT_HCA /
// ACCEPT_DEVICE readers.

#include <cstdlib>
#include <string>

#include "common/Capability.hpp"
#include "common/Topology.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;
using namespace acceptance::topology;

namespace accept_test {

namespace {

bool has(const std::string& s, const std::string& part) {
  return s.find(part) != std::string::npos;
}

// Two ACTIVE NICs, four GPUs: GPUs 0-1 nearest n0, GPUs 2-3 nearest n1. HIP
// and KFD ordinals agree.
EnvironmentInfo two_nic_env() {
  EnvironmentInfo env;
  for (const char* n : {"n0", "n1"}) {
    RdmaDevice d;
    d.device = n;
    d.port_state = "ACTIVE";
    env.rdma_devices.push_back(d);
  }
  env.gpu_count = 4;
  for (int i = 0; i < 4; ++i) {
    GpuDevice g;
    g.ordinal = i;
    g.hip_ordinal = i;
    env.gpu_devices.push_back(g);
    GpuNicAffinity a;
    a.gpu_ordinal = i;
    a.nearest_nic = i < 2 ? "n0" : "n1";
    env.gpu_nic_affinity.push_back(a);
  }
  return env;
}

void test_parse_kfd_bdf() {
  // Ruby (MI350X): GPU0 is 0000:08:00.0 and KFD prints location_id 2048.
  CHECK_EQ_STR(parse_kfd_bdf("2048", "0"), "0000:08:00.0", "bus 8, devfn 0");
  CHECK_EQ_STR(parse_kfd_bdf("49425", "1"), "0001:c1:02.1",
               "bus 0xc1, dev 2, fn 1, domain 1");
  CHECK_EQ_STR(parse_kfd_bdf("2048", ""), "0000:08:00.0", "no domain -> 0");
  CHECK_EQ_STR(parse_kfd_bdf("", "0"), "", "no location_id");
  CHECK_EQ_STR(parse_kfd_bdf("0x800", "0"), "", "not decimal");
  CHECK_EQ_STR(parse_kfd_bdf("65536", "0"), "", "above 16 bits");
}

void test_get_pci_shared_depth() {
  // Ruby: GPU0 and bnxt_re0 share a switch under root port 00:01.1; bnxt_re1
  // is under another root port.
  const std::string gpu0 =
      "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:03.0/"
      "0000:06:00.0/0000:07:00.0/0000:08:00.0";
  const std::string re0 =
      "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0";
  const std::string re1 =
      "/sys/devices/pci0000:00/0000:00:01.2/0000:11:00.0/0000:12:00.0/0000:13:00.0";
  CHECK(get_pci_shared_depth(gpu0, re0) == 5, "same switch: sys/devices/root/rp/switch");
  CHECK(get_pci_shared_depth(gpu0, re1) == 3, "same root complex only");
  CHECK(get_pci_shared_depth(gpu0, gpu0) == 9, "identical path");
  CHECK(get_pci_shared_depth(gpu0, "") == 0, "unresolved path");
}

void test_defaults() {
  const EnvironmentInfo env = two_nic_env();
  {
    const Pair s = select_pair(env, 0, "", -1);
    CHECK(s.gpu == 0 && s.gpu_source == "default", "local rank 0 -> GPU 0");
    CHECK_EQ_STR(s.hca, std::string("n0"), "GPU 0 -> its nearest NIC");
    CHECK_EQ_STR(s.hca_source, std::string("affinity"), "hca from affinity");
    CHECK(s.local, "nearest pair is local");
    CHECK_EQ_STR(s.note, std::string(""), "no fallback, no note");
  }
  {
    const Pair s = select_pair(env, 2, "", -1);
    CHECK(s.gpu == 2, "local rank 2 -> GPU 2");
    CHECK_EQ_STR(s.hca, std::string("n1"), "GPU 2 -> n1");
  }
  {
    const Pair s = select_pair(env, 7, "", -1);
    CHECK(s.gpu == 0 && s.gpu_source == "default", "non-visible local rank -> GPU 0");
    CHECK(has(s.note, "GPU 7 (local rank) is not visible"), s.note.c_str());
    CHECK_EQ_STR(s.hca, std::string("n0"), "then GPU 0's nearest NIC");
  }
}

void test_user_choices() {
  const EnvironmentInfo env = two_nic_env();
  {
    const Pair s = select_pair(env, 0, "", 3);
    CHECK(s.gpu == 3 && s.gpu_source == "user", "--device 3");
    CHECK_EQ_STR(s.hca, std::string("n1"), "--device 3 -> n1 by affinity");
    CHECK(s.local, "--device with its nearest NIC is local");
  }
  {
    const Pair s = select_pair(env, 0, "n1", -1);
    CHECK(s.gpu == 2 && s.gpu_source == "affinity", "--hca n1 -> lowest GPU near n1");
    CHECK(s.hca == "n1" && s.hca_source == "user", "--hca as given");
    CHECK(s.local, "--hca with its nearest GPU is local");
    CHECK_EQ_STR(s.note, std::string(""), "no note");
  }
  {
    const Pair s = select_pair(env, 0, "n1", 0);
    CHECK(s.gpu == 0 && s.hca == "n1", "--hca + --device both as given");
    CHECK(!s.local, "non-nearest pair is not local");
    CHECK(has(s.note, "not the nearest NIC to GPU 0 (nearest: n0)"), s.note.c_str());
  }
  {
    const Pair s = select_pair(env, 0, "mlx5_9", -1);
    CHECK_EQ_STR(s.hca, std::string("mlx5_9"), "absent --hca kept as given");
    CHECK(s.gpu == 0 && s.gpu_source == "default", "no GPU near it -> default");
    CHECK(has(s.error, "HCA mlx5_9 is not present"), s.error.c_str());
    CHECK(has(s.note, "no GPU is known to be nearest to mlx5_9"), s.note.c_str());
  }
  {
    const Pair s = select_pair(env, 0, "", 9);
    CHECK(s.gpu == 9 && s.gpu_source == "user", "non-visible --device kept");
    CHECK(has(s.error, "GPU 9 is not visible"), s.error.c_str());
    CHECK(s.hca == "n0" && s.hca_source == "default", "no affinity for it -> first ACTIVE");
  }
  {
    const Pair s = select_pair(env, 0, "mlx5_9", 9);
    CHECK(has(s.error, "GPU 9 is not visible on this host; HCA mlx5_9 is not present"),
          s.error.c_str());
  }
}

void test_inactive_and_ambiguous() {
  {
    EnvironmentInfo env = two_nic_env();
    env.rdma_devices[1].port_state = "DOWN";
    const Pair s = select_pair(env, 0, "", 2);
    CHECK(s.hca == "n0" && s.hca_source == "default", "DOWN nearest NIC -> first ACTIVE");
    CHECK(!s.local, "fallback NIC is not local");
    CHECK(has(s.note, "nearest NIC n1 to GPU 2 is not ACTIVE; using n0"), s.note.c_str());
  }
  {
    EnvironmentInfo env = two_nic_env();
    env.rdma_devices[0].port_state = "DOWN";
    const Pair s = select_pair(env, 0, "", -1);
    CHECK_EQ_STR(s.hca, std::string("n1"), "first ACTIVE skips a DOWN n0");
  }
  {
    EnvironmentInfo env = two_nic_env();
    env.gpu_nic_affinity[0].ambiguous = true;
    const Pair s = select_pair(env, 0, "", -1);
    CHECK(s.hca == "n0" && s.hca_source == "default", "tie -> first ACTIVE");
    CHECK(!s.local, "a tie never counts as local");
    CHECK(has(s.note, "can't tell NICs apart for GPU 0"), s.note.c_str());

    const Pair h = select_pair(env, 0, "n0", -1);
    CHECK(h.gpu == 1, "--hca skips a GPU whose nearest NIC is a tie");
  }
}

void test_no_affinity() {
  {
    // No KFD GPU list: the count decides visibility, and no note for the NIC.
    EnvironmentInfo env = two_nic_env();
    env.gpu_devices.clear();
    env.gpu_nic_affinity.clear();
    env.gpu_count = 2;
    const Pair s = select_pair(env, 1, "", -1);
    CHECK(s.gpu == 1, "visible by count");
    CHECK(s.hca == "n0" && s.hca_source == "default", "first ACTIVE NIC");
    CHECK_EQ_STR(s.note, std::string(""), "no GPU list -> no note");
    CHECK(has(select_pair(env, 0, "", 5).error, "GPU 5 is not visible"),
          "beyond the count is not visible");
  }
  {
    EnvironmentInfo env = two_nic_env();
    env.gpu_nic_affinity.clear();
    const Pair s = select_pair(env, 0, "", -1);
    CHECK(has(s.note, "no GPU-NIC affinity for GPU 0; using n0"), s.note.c_str());
  }
  {
    EnvironmentInfo env = two_nic_env();
    env.rdma_devices.clear();
    const Pair s = select_pair(env, 0, "", -1);
    CHECK(s.hca.empty() && s.hca_source.empty() && !s.local, "no NICs -> no HCA");
    CHECK(s.note.empty(), "no NICs -> no note");
    CHECK_EQ_STR(format_pair(s), std::string("GPU 0 (default), HCA none"),
                 "header without NICs");
  }
}

void test_format_pair() {
  const EnvironmentInfo env = two_nic_env();
  CHECK_EQ_STR(format_pair(select_pair(env, 0, "", -1)),
               std::string("GPU 0 (default), HCA n0 (affinity)"), "default header");
  CHECK_EQ_STR(format_pair(select_pair(env, 0, "n1", 2)),
               std::string("GPU 2 (user), HCA n1 (user)"), "user header");
}

void test_hip_ordinals() {
  // HIP_VISIBLE_DEVICES hid KFD GPU 0: HIP 0..2 are KFD 1..3.
  EnvironmentInfo env = two_nic_env();
  env.gpu_devices[0].hip_ordinal = -1;
  for (int i = 1; i < 4; ++i) env.gpu_devices[i].hip_ordinal = i - 1;
  const GpuNicAffinity* a = find_gpu_affinity(env, 0);
  CHECK(a != nullptr && a->gpu_ordinal == 1, "HIP 0 -> KFD 1's affinity");
  CHECK(find_gpu_affinity(env, 3) == nullptr, "HIP 3 does not exist");
  CHECK_EQ_STR(select_pair(env, 0, "", 1).hca, std::string("n1"),
               "HIP 1 (KFD 2) -> n1");
  CHECK(select_pair(env, 0, "n1", -1).gpu == 1, "--hca n1 -> HIP 1, not KFD 2");
  CHECK(select_pair(env, 0, "n0", -1).gpu == 0, "--hca n0 -> HIP 0 (KFD 1)");
  CHECK(has(select_pair(env, 3, "", -1).note, "GPU 3 (local rank) is not visible"),
        "local rank beyond the HIP GPUs");
}

void test_validate_lists() {
  CHECK(validate_lists("", "", 1).empty(), "nothing given");
  CHECK(validate_lists("a", "0", 2).empty(), "one entry for every rank");
  CHECK(validate_lists("a,b", "0,4", 2).empty(), "one entry per local rank");
  CHECK(has(validate_lists("a,b", "", 1), "--hca lists 2 entries"),
        "--hca longer than -ppn");
  CHECK(has(validate_lists("", "0,1,2", 2), "--device lists 3 entries"),
        "--device longer than -ppn");
  CHECK(has(validate_lists("", "mlx5_0", 1), "use --hca for the RDMA device"),
        "an NIC name in --device");
  CHECK(!validate_lists("", "-1", 1).empty(), "negative --device");
  CHECK(!validate_lists("", "0,x", 2).empty(), "non-numeric entry");
}

void test_get_rank_entry() {
  CHECK_EQ_STR(get_rank_entry("", 0), std::string(""), "empty list");
  CHECK_EQ_STR(get_rank_entry("a", 3), std::string("a"), "one entry for all");
  CHECK_EQ_STR(get_rank_entry("a,b", 0), std::string("a"), "local rank 0");
  CHECK_EQ_STR(get_rank_entry("a,b", 1), std::string("b"), "local rank 1");
  CHECK_EQ_STR(get_rank_entry("a,b", -1), std::string("a"), "no local rank -> first");
}

void test_env_readers() {
  ::unsetenv("ACCEPT_HCA");
  ::unsetenv("ACCEPT_DEVICE");
  ::unsetenv("ACCEPT_LOCAL_RANK");
  CHECK(get_rank_hca().empty() && get_rank_gpu() == 0, "unset -> no HCA, GPU 0");

  ::setenv("ACCEPT_DEVICE", "bnxt_re0", 1);
  CHECK(get_rank_gpu() == 0, "malformed ACCEPT_DEVICE -> GPU 0");

  const EnvironmentInfo env = two_nic_env();
  ::setenv("ACCEPT_HCA", "n1", 1);
  ::setenv("ACCEPT_DEVICE", "3", 1);
  ::setenv("ACCEPT_LOCAL_RANK", "1", 1);
  CHECK(get_rank_hca() == "n1" && get_rank_gpu() == 3,
        "readers return the published pair");
  const Pair s = get_rank_pair(env);
  CHECK(s.hca == "n1" && s.gpu == 3 && s.local, "get_rank_pair reads all three");

  ::unsetenv("ACCEPT_HCA");
  ::unsetenv("ACCEPT_DEVICE");
  CHECK(get_rank_pair(env).gpu == 1, "ACCEPT_LOCAL_RANK picks the default GPU");
  ::unsetenv("ACCEPT_LOCAL_RANK");
}

}  // namespace

void run_topology_tests() {
  test_parse_kfd_bdf();
  test_get_pci_shared_depth();
  test_defaults();
  test_user_choices();
  test_inactive_and_ambiguous();
  test_no_affinity();
  test_format_pair();
  test_hip_ordinals();
  test_validate_lists();
  test_get_rank_entry();
  test_env_readers();
}

}  // namespace accept_test
