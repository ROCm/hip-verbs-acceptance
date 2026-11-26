/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_capability — tests for the pure, HIP-free helpers behind the ROCm
// version resolution (common/Capability.*):
//   * decode_hip_version(int) -- HIP_VERSION integer -> "major.minor.patch".
//   * find_info_version(start_dir, max_levels) -- walk-up-for-.info/version,
//     exercised against a temp directory tree created (and removed) here.
// and for the NIC vendor detection:
//   * probe_rdma_devices(ib_root) -- PCI IDs and vendor per device, against a
//     fake /sys/class/infiniband tree.
//   * select_rank_hca / mixed_vendor_warning -- the rank's HCA picks env.vendor.
// All are plain filesystem/arithmetic code with no ACCEPT_HAVE_HIP
// dependency, so they run in every unit_tests build (synthetic and full).

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <sys/stat.h>  // mkfifo

#include "common/Capability.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

namespace accept_test {

namespace {

void test_decode_hip_version() {
  CHECK_EQ_STR(decode_hip_version(60342134), "6.3.42134",
               "decode_hip_version: 60342134 -> 6.3.42134");
  CHECK_EQ_STR(decode_hip_version(60000000), "6.0.0",
               "decode_hip_version: round major -> 6.0.0");
  CHECK_EQ_STR(decode_hip_version(100100000), "10.1.0",
               "decode_hip_version: two-digit major -> 10.1.0");
  CHECK_EQ_STR(decode_hip_version(0), "",
               "decode_hip_version: zero -> empty (no HIP)");
  CHECK_EQ_STR(decode_hip_version(-1), "",
               "decode_hip_version: negative -> empty");
}

// Write `content` to `path`, creating parent directories as needed.
void write_file(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << content;
}

void test_find_info_version() {
  // Scratch tree, removed in its entirety at the end of this function.
  char tmpl[] = "/tmp/accept_unit_capability_XXXXXX";
  const char* mk = ::mkdtemp(tmpl);
  CHECK(mk != nullptr, "find_info_version: mkdtemp scratch dir");
  if (mk == nullptr) return;
  const std::filesystem::path root = mk;

  // Case 1: .info/version one level above a "lib" dir (the common ROCm
  // install layout: <root>/lib/libamdhip64.so, <root>/.info/version).
  const std::filesystem::path case1_root = root / "case1";
  const std::filesystem::path case1_lib = case1_root / "lib";
  write_file(case1_root / ".info" / "version", "6.4.2-120\n");
  std::filesystem::create_directories(case1_lib);

  const auto hit0 = find_info_version(case1_lib.string(), 0);
  CHECK(!hit0.has_value(),
        "find_info_version: level 0 only checks lib/ itself -> not found");
  const auto hit1 = find_info_version(case1_lib.string(), 1);
  CHECK(hit1.has_value() && *hit1 == "6.4.2-120",
        "find_info_version: level 1 finds the parent's .info/version");

  // Case 2: the helper honours max_levels for deeper layouts too
  // (lib -> z -> y -> x, i.e. 3 parent hops from lib to the hit). The runtime
  // lookup itself only goes 1 level up.
  const std::filesystem::path case2_hit_dir = root / "case2" / "x";
  const std::filesystem::path case2_lib = case2_hit_dir / "y" / "z" / "lib";
  write_file(case2_hit_dir / ".info" / "version", "10.2.0a20261013\n");
  std::filesystem::create_directories(case2_lib);

  const auto hit2 = find_info_version(case2_lib.string(), 2);
  CHECK(!hit2.has_value(),
        "find_info_version: 2 levels is one short of the case2 hit");
  const auto hit3 = find_info_version(case2_lib.string(), 3);
  CHECK(hit3.has_value() && *hit3 == "10.2.0a20261013",
        "find_info_version: 3 levels finds a deeper .info/version, trailing "
        "newline stripped");

  // Case 3: nothing to find within the search depth -> nullopt, no throw.
  const std::filesystem::path case3_lib = root / "case3" / "lib";
  std::filesystem::create_directories(case3_lib);
  const auto hit_none = find_info_version(case3_lib.string(), 3);
  CHECK(!hit_none.has_value(),
        "find_info_version: no .info/version anywhere in range -> nullopt");

  // Case 4: a start_dir that does not exist at all -> nullopt, no throw/crash.
  const auto hit_missing =
      find_info_version((root / "does" / "not" / "exist").string(), 3);
  CHECK(!hit_missing.has_value(),
        "find_info_version: nonexistent start_dir -> nullopt");

  // Case 5: degenerate inputs are rejected cleanly.
  CHECK(!find_info_version("", 3).has_value(),
        "find_info_version: empty start_dir -> nullopt");
  CHECK(!find_info_version(case1_lib.string(), -1).has_value(),
        "find_info_version: negative max_levels -> nullopt");

  // Case 6: content that is not a version string is ignored, so arbitrary
  // file content never reaches the report.
  const std::filesystem::path case6 = root / "case6";
  write_file(case6 / ".info" / "version", "not a version\n");
  CHECK(!find_info_version(case6.string(), 0).has_value(),
        "find_info_version: non-version content -> nullopt");

  // Case 7: a FIFO in place of the file is skipped without blocking.
  const std::filesystem::path case7 = root / "case7";
  std::filesystem::create_directories(case7 / ".info");
  CHECK(::mkfifo((case7 / ".info" / "version").c_str(), 0600) == 0,
        "find_info_version: mkfifo scratch FIFO");
  CHECK(!find_info_version(case7.string(), 0).has_value(),
        "find_info_version: FIFO -> nullopt, no hang");

  // Case 8: a directory in place of the file is skipped.
  const std::filesystem::path case8 = root / "case8";
  std::filesystem::create_directories(case8 / ".info" / "version");
  CHECK(!find_info_version(case8.string(), 0).has_value(),
        "find_info_version: directory -> nullopt");

  // Cleanup: remove the whole scratch tree so the test leaves nothing behind.
  std::error_code rm_ec;
  std::filesystem::remove_all(root, rm_ec);
  CHECK(!rm_ec, "find_info_version: scratch tree cleaned up");
}

// A fake /sys/class/infiniband/<dev> entry; an empty id leaves its file out.
void write_ib_device(const std::filesystem::path& ib_root, const std::string& dev,
                     const std::string& vendor, const std::string& device,
                     const std::string& subsystem_device) {
  const std::filesystem::path pci = ib_root / dev / "device";
  std::filesystem::create_directories(pci);
  if (!vendor.empty()) write_file(pci / "vendor", vendor + "\n");
  if (!device.empty()) write_file(pci / "device", device + "\n");
  if (!subsystem_device.empty())
    write_file(pci / "subsystem_device", subsystem_device + "\n");
}

void test_nic_vendor_from_pci() {
  CHECK_EQ_STR(nic_vendor_from_pci("0x15b3"), "connectx", "pci vendor: Mellanox");
  CHECK_EQ_STR(nic_vendor_from_pci("0x14e4"), "thor", "pci vendor: Broadcom");
  CHECK_EQ_STR(nic_vendor_from_pci("0x1dd8"), "ionic", "pci vendor: Pensando");
  CHECK_EQ_STR(nic_vendor_from_pci("0x8086"), "unknown", "pci vendor: other");
  CHECK_EQ_STR(nic_vendor_from_pci(""), "unknown", "pci vendor: unreadable");
}

void test_probe_rdma_devices() {
  char tmpl[] = "/tmp/accept_unit_ib_XXXXXX";
  const char* mk = ::mkdtemp(tmpl);
  CHECK(mk != nullptr, "probe_rdma_devices: mkdtemp scratch dir");
  if (mk == nullptr) return;
  const std::filesystem::path root = mk;

  write_ib_device(root, "rdma0", "0x1dd8", "0x1002", "0x5201");
  write_ib_device(root, "bnxt_re0", "0x14e4", "0x1760", "0x5208");
  write_ib_device(root, "mlx5_0", "0x15b3", "0x1021", "0x0023");
  write_ib_device(root, "rocep65s0f0", "0x15b3", "0x101d", "");
  write_ib_device(root, "nopci0", "", "", "");
  write_ib_device(root, "garbage0", "not-hex", "0x12345", "");
  write_ib_device(root, "intel0", "0x8086", "0x1593", "");

  const std::vector<RdmaDevice> devs = probe_rdma_devices(root.string());
  CHECK(devs.size() == 7, "probe_rdma_devices: all 7 devices enumerated");
  EnvironmentInfo env;
  env.rdma_devices = devs;
  auto dev = [&](const char* name) -> RdmaDevice {
    const RdmaDevice* d = find_rdma_device(env, name);
    CHECK(d != nullptr, (std::string("probe_rdma_devices: found ") + name).c_str());
    return d ? *d : RdmaDevice{};
  };

  const RdmaDevice rdma0 = dev("rdma0");
  CHECK_EQ_STR(rdma0.vendor, "ionic", "renamed rdma0 with 0x1dd8 -> ionic");
  CHECK_EQ_STR(rdma0.vendor_source, "pci", "rdma0 vendor_source");
  CHECK_EQ_STR(rdma0.pci_subsystem_device, "0x5201", "rdma0 subsystem_device");

  const RdmaDevice bnxt = dev("bnxt_re0");
  CHECK_EQ_STR(bnxt.vendor, "thor", "bnxt_re0 with 0x14e4 -> thor");
  CHECK_EQ_STR(bnxt.pci_vendor, "0x14e4", "bnxt_re0 pci_vendor");
  CHECK_EQ_STR(bnxt.pci_device, "0x1760", "bnxt_re0 pci_device (Thor 2)");

  CHECK_EQ_STR(dev("mlx5_0").vendor, "connectx", "mlx5_0 with 0x15b3 -> connectx");

  const RdmaDevice rocep = dev("rocep65s0f0");
  CHECK_EQ_STR(rocep.vendor, "connectx",
               "rocep65s0f0 with 0x15b3 -> connectx (the name does not matter)");
  CHECK_EQ_STR(rocep.pci_subsystem_device, "", "missing subsystem_device -> empty");

  const RdmaDevice nopci = dev("nopci0");
  CHECK_EQ_STR(nopci.vendor, "unknown", "missing vendor file -> unknown");
  CHECK_EQ_STR(nopci.vendor_source, "none", "missing vendor file -> source none");

  const RdmaDevice garbage = dev("garbage0");
  CHECK_EQ_STR(garbage.vendor, "unknown", "non-hex vendor file -> unknown");
  CHECK_EQ_STR(garbage.vendor_source, "none", "non-hex vendor file -> source none");
  CHECK_EQ_STR(garbage.pci_device, "", "over-long device id -> empty");

  const RdmaDevice intel = dev("intel0");
  CHECK_EQ_STR(intel.vendor, "unknown", "unmapped vendor id -> unknown");
  CHECK_EQ_STR(intel.vendor_source, "pci", "unmapped vendor id -> source pci");

  CHECK(probe_rdma_devices((root / "missing").string()).empty(),
        "probe_rdma_devices: missing root -> no devices");

  std::error_code rm_ec;
  std::filesystem::remove_all(root, rm_ec);
  CHECK(!rm_ec, "probe_rdma_devices: scratch tree cleaned up");
}

RdmaDevice fake_device(const std::string& name, const std::string& vendor) {
  RdmaDevice d;
  d.device = name;
  d.vendor = vendor;
  return d;
}

void test_select_rank_hca() {
  EnvironmentInfo env;
  env.rdma_devices = {fake_device("bnxt_re0", "thor"), fake_device("mlx5_0", "connectx"),
                      fake_device("rdma0", "ionic"), fake_device("rdma1", "ionic")};

  select_rank_hca(env, "rdma1");
  CHECK_EQ_STR(env.rdma_devices.front().device, "rdma1", "rank HCA moved to the front");
  CHECK_EQ_STR(env.vendor, "ionic", "mixed vendors: the rank HCA's vendor wins");
  CHECK_EQ_STR(env.rdma_prefix, "", "renamed rank HCA -> empty rdma_prefix");
  CHECK_EQ_STR(mixed_vendor_warning(env),
               "RDMA devices from more than one NIC vendor on this node: ionic: rdma1 "
               "rdma0; thor: bnxt_re0; connectx: mlx5_0; using ionic (the vendor of "
               "this rank's HCA rdma1)",
               "mixed vendors: warning lists the devices per vendor");

  select_rank_hca(env, "bnxt_re0");
  CHECK_EQ_STR(env.vendor, "thor", "re-selecting bnxt_re0 -> thor");
  CHECK_EQ_STR(env.rdma_prefix, "bnxt_re", "bnxt_re0 -> rdma_prefix bnxt_re");

  select_rank_hca(env, "not_here");
  CHECK_EQ_STR(env.rdma_devices.front().device, "bnxt_re0",
               "absent HCA: device order unchanged");
  CHECK_EQ_STR(env.vendor, "thor", "absent HCA: vendor of the first device");

  EnvironmentInfo uniform;
  uniform.rdma_devices = {fake_device("rdma0", "ionic"), fake_device("rdma1", "ionic")};
  select_rank_hca(uniform, "");
  CHECK_EQ_STR(uniform.vendor, "ionic", "uniform vendors -> ionic");
  CHECK_EQ_STR(mixed_vendor_warning(uniform), "", "uniform vendors -> no warning");

  EnvironmentInfo none;
  none.vendor = "stale";
  none.rdma_prefix = "stale";
  select_rank_hca(none, "");
  CHECK_EQ_STR(none.vendor, "unknown", "no devices -> unknown");
  CHECK_EQ_STR(none.rdma_prefix, "", "no devices -> empty rdma_prefix");
  CHECK_EQ_STR(mixed_vendor_warning(none), "", "no devices -> no warning");
}

}  // namespace

void run_capability_tests() {
  test_decode_hip_version();
  test_find_info_version();
  test_nic_vendor_from_pci();
  test_probe_rdma_devices();
  test_select_rank_hca();
}

}  // namespace accept_test
