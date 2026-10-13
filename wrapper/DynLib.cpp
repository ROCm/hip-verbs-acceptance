/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "wrapper/DynLib.hpp"

#include <dlfcn.h>

namespace acceptance {
namespace wrap {

int load_library(lib_info_t& info) {
  const int flags =
      info.dlopen_flags != 0 ? info.dlopen_flags : (RTLD_LAZY | RTLD_GLOBAL);
  info.handle = ::dlopen(info.path.c_str(), flags);
  if (!info.handle) {
    return WRAP_LOAD_DLOPEN_ERROR;
  }

  void** ops = reinterpret_cast<void**>(info.ops);
  const auto& fn_names = info.fn_names;
  info.missing.clear();
  // Tolerant resolution: a missing symbol is NOT a load failure. The ops-table
  // member stays null and is recorded in `missing`; any test that later reaches
  // for it skips via checked_op/MissingSymbol. This lets the suite run on an
  // older runtime that lacks newer APIs without crashing -- only the dependent
  // tests skip. The single hard failure is the library not opening at all
  // (handled above), which the per-category gate turns into a clean skip.
  for (size_t i = 0; i < fn_names.size(); ++i) {
    ops[i] = ::dlsym(info.handle, fn_names[i].c_str());
    if (!ops[i]) {
      info.missing.push_back(fn_names[i]);
    }
  }
  return WRAP_LOAD_OK;
}

void close_library(lib_info_t& info) {
  if (info.handle) {
    ::dlclose(info.handle);
    info.handle = nullptr;
  }
}

}  // namespace wrap
}  // namespace acceptance
