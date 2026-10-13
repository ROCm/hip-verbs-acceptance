/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Basic synthetic tests: deterministic pass/fail/skip outcomes used to validate
// the EXEC path and (later) the reduce + classification pipeline.

#include "common/TestRegistry.hpp"

namespace acceptance {

ACCEPT_TEST(syn_pass, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  return pass();
}

ACCEPT_TEST(syn_fail, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  return fail("intentional failure");
}

ACCEPT_TEST(syn_skip, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  return skip("intentional skip");
}

}  // namespace acceptance
