/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Two-rank synthetic tests (Step 5). Topology TwoNode: these run IN-PROCESS in
// each worker under the shared SocketBootstrap (they are cooperative, not
// crash-prone, and the forked --exec child cannot share the live socket). Each
// rank reports its HONEST local outcome; the Reporter reconciles across ranks
// (ARCHITECTURE.md §7.3). There is deliberately NO cross-rank pass/fail
// consensus here (§4 non-goal).

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <unistd.h>

#include "common/Bootstrap.hpp"
#include "common/TestRegistry.hpp"

namespace acceptance {

// syn2_barrier: exercises the real SocketBootstrap across two processes. Needs
// exactly 2 ranks; barrier()s, then exchanges an int with the peer and verifies
// it received the peer's rank (1 - own rank). Both ranks pass on success.
ACCEPT_TEST(syn2_barrier, "synthetic", Topology::TwoNode, Kind::Functional, 30) {
  if (ctx.size != 2 || ctx.boot == nullptr) {
    return skip("needs 2 ranks");
  }
  ctx.boot->barrier();

  const int32_t mine = ctx.rank;
  int32_t peer = -1;
  ctx.boot->exchange(1 - ctx.rank, &mine, &peer, sizeof(mine));

  if (peer != (1 - ctx.rank)) {
    return fail("peer value mismatch: got " + std::to_string(peer) +
                ", expected " + std::to_string(1 - ctx.rank));
  }
  return pass();
}

// syn2_r0fail_r1pass: rank 0 fails, rank 1 passes. Reconciles to fail (§7.3:
// any rank fail -> fail).
ACCEPT_TEST(syn2_r0fail_r1pass, "synthetic", Topology::TwoNode, Kind::Functional,
            30) {
  if (ctx.size != 2) return skip("needs 2 ranks");
  if (ctx.rank == 0) return fail("rank0 intentional");
  return pass();
}

// syn2_r0pass_r1passive: rank 0 passes, rank 1 is the passive/idle peer
// (returns none_()). Reconciles to pass (None is the lattice identity);
// reduced_from {"0":"pass","1":"none"}.
ACCEPT_TEST(syn2_r0pass_r1passive, "synthetic", Topology::TwoNode,
            Kind::Functional, 30) {
  if (ctx.size != 2) return skip("needs 2 ranks");
  if (ctx.rank == 0) return pass();
  return none_();
}

// syn2_hang: both ranks spin forever (never call pass()/fail()). Opt-in via
// ACCEPT_SYN_HANG=1 -- selected by default it would otherwise hang any run
// that includes it, which is the opposite of a harmless self-test fixture.
// Exercises the worker-side --deadline-sec watchdog: TwoNode tests
// run IN-PROCESS on the worker's main thread (Runner::run_in_process has no
// per-test timeout, unlike the isolated --exec path), so nothing but the
// worker's own deadline watchdog can ever end this.
ACCEPT_TEST(syn2_hang, "synthetic", Topology::TwoNode, Kind::Functional, 30) {
  if (ctx.size != 2) return skip("needs 2 ranks");
  if (std::getenv("ACCEPT_SYN_HANG") == nullptr) {
    return skip("ACCEPT_SYN_HANG not set");
  }
  for (;;) {
    ::sleep(1);
  }
}

}  // namespace acceptance
