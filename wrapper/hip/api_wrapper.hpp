/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once

#include <string>

#include "wrapper/DynLib.hpp"

namespace acceptance {
namespace hipw {

// The generic dlopen loader now lives in acceptance::wrap (wrapper/DynLib.hpp).
// Re-export the pieces the HIP ops tables and existing call sites reference under
// the historical hipw:: names so nothing downstream had to change.
using wrap::checked_op;
using wrap::close_library;
using wrap::lib_info_t;
using wrap::load_library;
using wrap::MissingSymbol;

// Back-compat aliases for the old return-code spellings used by the HIP ops-table
// init functions.
#define HIPW_LOAD_LB_SUCCESS WRAP_LOAD_OK
#define HIPW_LOAD_LB_DLOPEN_ERROR WRAP_LOAD_DLOPEN_ERROR

struct InitResult {
  bool hip_lib_loaded = false;
  bool hsa_lib_loaded = false;
  // hiprtc (runtime kernel compilation) is only needed by kernel-launch tests;
  // its absence is not fatal (those tests skip), so it is NOT part of ok().
  bool hiprtc_lib_loaded = false;
  bool ok() const { return hip_lib_loaded && hsa_lib_loaded; }
};

InitResult& init_result();

// True when `category` names a HIP test ("hip/...") but libamdhip64.so itself
// did not load (no ROCm on this host at all). The runner skips the whole
// category with one clean reason instead of letting each test fault on a null
// ops table. Per-symbol gaps on a runtime that DID load are handled separately
// by checked_op/MissingSymbol at the individual call sites.
inline bool hip_category_unavailable(const std::string& category) {
  return category.rfind("hip", 0) == 0 && !init_result().hip_lib_loaded;
}

void api_wrappers_init();
void api_wrappers_fini();

// Suite entry point (main): oneCCL global_data::init() -> api_wrappers_init().
const InitResult& init();

}  // namespace hipw
}  // namespace acceptance
