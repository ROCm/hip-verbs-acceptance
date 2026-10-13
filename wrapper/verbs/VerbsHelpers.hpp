/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once
// VerbsHelpers — host-side helpers for the Verbs acceptance tests. The ibv_*
// symbols resolve through the dynamic wrapper (IbvApi.hpp), so this compiles
// under plain g++ against the vendored rdma-core headers with no libibverbs-dev.

#include "wrapper/verbs/IbvApi.hpp"

#include <cerrno>
#include <cstring>
#include <string>

#include "common/Perf.hpp"

// IBV_CHECK — evaluate a verbs call that returns 0 on success / errno on
// failure; on non-zero, return fail() from the enclosing ACCEPT_TEST body with
// the call text + strerror. Under perf mode (PerfProbeScope active) it also
// records the call's latency; otherwise PerfTimer is a no-op. Mirrors HIP_CHECK.
#define IBV_CHECK(expr)                                                        \
  do {                                                                         \
    ::acceptance::PerfTimer _ibv_timer(#expr);                                 \
    int _ibv_rc = (expr);                                                      \
    _ibv_timer.stop();                                                         \
    if (_ibv_rc != 0) {                                                        \
      return ::acceptance::fail(std::string(#expr) + ": " +                    \
                                std::strerror(_ibv_rc > 0 ? _ibv_rc : errno)); \
    }                                                                          \
  } while (0)

// IBV_CHECK_PTR — for constructor calls that return a pointer / null on failure
// (errno set). The pointer is created before the macro, so per-call latency for
// creates is captured with PerfLoop regions instead (see foundation.cc). On
// null, return fail() with the call text + strerror(errno).
#define IBV_CHECK_PTR(ptr, what)                                               \
  do {                                                                         \
    if ((ptr) == nullptr) {                                                    \
      return ::acceptance::fail(std::string(what) + " failed: " +             \
                                std::strerror(errno));                         \
    }                                                                          \
  } while (0)
