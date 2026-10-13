/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Shared per-rank setup gate for the two-node verbs/HIP tests. Each rank reports
// how its own setup went; reconcile_gate() allgathers those reports so every
// rank takes the SAME follow-up decision and stays in lock-step on the
// collective sequence (a rank that skipped locally must not keep barrier()-ing
// against peers that continued). This is test-internal setup coordination, not a
// harness-level pass/fail consensus.

#ifdef ACCEPT_HAVE_IBVERBS

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common/Bootstrap.hpp"

namespace acceptance {
namespace verbsx {

// Per-rank setup severity, ordered so the worst outcome wins: Ok < Skip < Fail.
enum class Gate : uint8_t { Ok = 0, Skip = 1, Fail = 2 };

// Returns the worst severity reported by any rank. Works for any rank count.
inline Gate reconcile_gate(Bootstrap* b, Gate mine) {
  const int n = b->size();
  const uint8_t self = static_cast<uint8_t>(mine);
  std::vector<uint8_t> all(static_cast<size_t>(n), 0);
  b->allgather(&self, all.data(), 1);
  Gate worst = Gate::Ok;
  for (int i = 0; i < n; ++i) worst = std::max(worst, static_cast<Gate>(all[i]));
  return worst;
}

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
