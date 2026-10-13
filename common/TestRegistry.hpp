/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// TestRegistry — compile-time test catalog. Tests register themselves via static
// initializers (the ACCEPT_TEST macro). The backing store is a Meyers singleton
// to dodge static-init-order issues across translation units.

#include <string>
#include <vector>

#include "common/Bootstrap.hpp"
#include "common/TestResult.hpp"
#include "common/Types.hpp"

namespace acceptance {

// Per-invocation context handed to each test body.
struct TestContext {
  int        rank = 0;
  int        size = 1;
  Bootstrap* boot = nullptr;  // null in Step 1-4; real in Step 5
};

using TestFn = TestResult (*)(TestContext&);

struct TestMeta {
  std::string id;
  std::string category;
  Topology    topology    = Topology::Single;
  Kind        kind        = Kind::Functional;
  int         timeout_sec = 30;
  TestFn      fn          = nullptr;
};

struct TestRegistry {
  static bool                         add(const TestMeta& m);  // returns true
  static const std::vector<TestMeta>& all();
  static const TestMeta*              find(const std::string& id);
};

// Registration macro — declares fn, registers via static init, then opens body.
#define ACCEPT_TEST(ID, CATEGORY, TOPO, KIND, TIMEOUT)                         \
  static ::acceptance::TestResult ID##_fn(::acceptance::TestContext&);                 \
  [[maybe_unused]] static const bool ID##_reg =                               \
      ::acceptance::TestRegistry::add({#ID, CATEGORY, TOPO, KIND, TIMEOUT,         \
                                   &ID##_fn});                                 \
  static ::acceptance::TestResult ID##_fn(::acceptance::TestContext& ctx)

// Body helpers. id/duration_ms are filled in by the EXEC runner, not the body;
// the remaining members keep their in-class defaults (see TestResult), so the
// designated-init below initializes every field and needs no diagnostic pragma.
inline TestResult pass()                { return {.status = Status::Pass}; }
inline TestResult fail(std::string why) { return {.status = Status::Fail, .reason = std::move(why)}; }
inline TestResult skip(std::string why) { return {.status = Status::Skip, .reason = std::move(why)}; }
inline TestResult none_()               { return {.status = Status::None}; }

}  // namespace acceptance
