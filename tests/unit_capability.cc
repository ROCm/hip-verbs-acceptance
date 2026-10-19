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
// Both helpers are plain filesystem/arithmetic code with no ACCEPT_HAVE_HIP
// dependency, so they run in every unit_tests build (synthetic and full).

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

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

  // Case 2: deeper nesting, right at the documented 3-level search depth
  // (lib -> z -> y -> x, i.e. 3 parent hops from lib to the hit).
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

  // Cleanup: remove the whole scratch tree so the test leaves nothing behind.
  std::error_code rm_ec;
  std::filesystem::remove_all(root, rm_ec);
  CHECK(!rm_ec, "find_info_version: scratch tree cleaned up");
}

}  // namespace

void run_capability_tests() {
  test_decode_hip_version();
  test_find_info_version();
}

}  // namespace accept_test
