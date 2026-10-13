/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Classification-fixture synthetic tests (Step 3). These exist to realize the
// rows of the validation matrix that the basic/failure fixtures do not cover:
// KNOWN_FAILURE (reproduced + "possibly fixed"), NA (synthesized, never run),
// UNCLASSIFIED (absent from policy), and the SIGTERM-ignoring hang that forces
// the watchdog's SIGTERM -> grace -> SIGKILL escalation path.

#include <csignal>   // signal, SIG_IGN
#include <unistd.h>  // pause

#include "common/TestRegistry.hpp"

namespace acceptance {

// KNOWN_FAILURE (expected_status="fail"): fails as expected -> gate passes
// (bug reproduced).
ACCEPT_TEST(syn_known_failure_fail, "synthetic", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return fail("known failure behavior");
}

// KNOWN_FAILURE (expected_status="fail"): unexpectedly passes -> Reporter flags
// it "possibly fixed" (gate still passes).
ACCEPT_TEST(syn_known_failure_pass, "synthetic", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  return pass();
}

// NA for the synthetic vendor: the worker skips it and the Reporter synthesizes
// the not-applicable row. The body is never executed; pass() is a placeholder.
ACCEPT_TEST(syn_na, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  return pass();
}

// Absent from classification.json's synthetic column (and inherit_defaults is
// false) -> resolves to UNCLASSIFIED, recorded + flagged. Gate not affected.
ACCEPT_TEST(syn_unclassified, "synthetic", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  return pass();
}

// Ignores SIGTERM, then blocks forever. The watchdog's first stage (SIGTERM)
// has no effect, so it must escalate through the grace period to SIGKILL (the
// path syn_hang does not exercise, since pause() is interrupted by SIGTERM).
// timeout_sec=2 keeps the watchdog runtime short. Result: fail / timeout.
ACCEPT_TEST(syn_hang_ignore_sigterm, "synthetic", Topology::Single,
            Kind::Functional, 2) {
  (void)ctx;
  ::signal(SIGTERM, SIG_IGN);
  for (;;) ::pause();  // only SIGKILL can stop this
  return fail("unreachable");
}

}  // namespace acceptance
