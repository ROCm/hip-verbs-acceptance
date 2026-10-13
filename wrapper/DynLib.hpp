/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// DynLib — backend-neutral dlopen/dlsym ops-table loader shared by every
// runtime wrapper (HIP, HSA, hiprtc, ibverbs, ...). The suite builds on a node
// with no vendor toolchain by compiling against vendored
// headers and resolving each entry point at run time through this loader; a
// missing symbol becomes a clean Skip at the test boundary (checked_op /
// MissingSymbol) rather than a link failure or a null-pointer call.

#include <string>
#include <vector>

namespace acceptance {
namespace wrap {

#define WRAP_LOAD_OK 0
#define WRAP_LOAD_DLOPEN_ERROR 1

// One dynamically loaded library and its resolved ops table. `ops` points at a
// backend-specific struct of function pointers laid out in the SAME order as
// `fn_names`; load_library() fills it slot by slot via dlsym.
struct lib_info_t {
  std::string path;
  void* handle = nullptr;
  void* ops = nullptr;
  std::vector<std::string> fn_names;
  // Symbols absent at load time (dlsym returned null). The load still succeeds
  // as long as the library itself opened; each missing member stays null and any
  // test that reaches for it skips at run time via checked_op / MissingSymbol.
  // Kept for diagnostics/reporting.
  std::vector<std::string> missing;
  // dlopen flags; 0 => loader default (RTLD_LAZY | RTLD_GLOBAL). The verbs
  // wrapper overrides to RTLD_LOCAL to avoid exposing ibv_* globally.
  int dlopen_flags = 0;
};

// Thrown by checked_op() when a test reaches for a wrapper symbol the running
// runtime did not provide (dlsym left the ops-table member null). Caught at the
// test-invocation boundary (Runner::run_in_process / main) and turned into a
// clean Skip. `name` points at a string literal with static storage duration.
struct MissingSymbol {
  const char* name;
};

// Guard wrapped around every ops-table function pointer at its call site (via
// the per-symbol #defines). Returns the pointer when present; throws
// MissingSymbol BEFORE the call when absent, so we never call through null.
template <class T>
inline T checked_op(T fn, const char* name) {
  if (fn == nullptr) throw MissingSymbol{name};
  return fn;
}

// dlopen the library and resolve every fn_names[i] into ops[i]. Missing symbols
// are tolerated (recorded in info.missing); the only hard failure is the library
// not opening at all. Returns WRAP_LOAD_OK or WRAP_LOAD_DLOPEN_ERROR.
int load_library(lib_info_t& info);
void close_library(lib_info_t& info);

}  // namespace wrap
}  // namespace acceptance
