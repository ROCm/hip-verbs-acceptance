/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// unit_harness — the tiny shared CHECK harness for the framework-free unit tests
// (no GoogleTest). Counters live in unit_reduce.cc
// (which owns main()); every other unit TU declares its entry point here and
// bumps the same counters via CHECK / CHECK_EQ. The process exits non-zero if any
// check fails (see unit_reduce.cc::main), keeping the target wired into ctest.

#include <cstdio>
#include <string>

namespace accept_test {

// Shared counters (defined in unit_reduce.cc).
extern int g_failures;
extern int g_checks;

// Boolean assertion with a message. Counts every check; records a failure and
// prints file:line on a false condition.
#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    ++::accept_test::g_checks;                                              \
    if (!(cond)) {                                                          \
      ++::accept_test::g_failures;                                          \
      std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                       \
  } while (0)

// Equality assertion that prints got vs want (both must be std::string or
// convertible to it) on mismatch. Asserts the real value, never a tautology.
inline void check_eq_str(const std::string& got, const std::string& want,
                         const char* msg, const char* file, int line) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    std::fprintf(stderr, "FAIL [%s:%d] %s: got=\"%s\" want=\"%s\"\n", file, line,
                 msg, got.c_str(), want.c_str());
  }
}

#define CHECK_EQ_STR(got, want, msg) \
  ::accept_test::check_eq_str((got), (want), (msg), __FILE__, __LINE__)

// Test entry points implemented in their own translation units.
void run_policy_tests();
void run_resultio_tests();
void run_ssh_tests();
void run_launch_tests();
void run_topology_tests();
void run_perf_tests();
void run_capability_tests();

}  // namespace accept_test
