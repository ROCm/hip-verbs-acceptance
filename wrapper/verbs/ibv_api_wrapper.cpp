/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "wrapper/verbs/ibv_api_wrapper.hpp"

#include <dlfcn.h>

namespace acceptance {
namespace ibvw {

ibv_lib_ops_t ibv_lib_ops;

namespace {
wrap::lib_info_t g_ibv_lib_info;
bool g_loaded = false;
}  // namespace

bool ibv_api_init() {
  g_ibv_lib_info.ops = &ibv_lib_ops;
  g_ibv_lib_info.fn_names = ibv_fn_names;
  // dlopen the runtime soname, not the -dev .so symlink: the login-node build
  // that produced this binary need not have libibverbs-dev installed, and the
  // GPU node only ships the versioned runtime library.
  g_ibv_lib_info.path = "libibverbs.so.1";
  // RTLD_LOCAL: don't expose ibv_* in the global namespace (avoids clashes if
  // two libibverbs versions coexist). Providers load via ibv_open_device.
  g_ibv_lib_info.dlopen_flags = RTLD_LAZY | RTLD_LOCAL;
  const int err = wrap::load_library(g_ibv_lib_info);
  g_loaded = (err == WRAP_LOAD_OK);
  return g_loaded;
}

void ibv_api_fini() {
  wrap::close_library(g_ibv_lib_info);
  g_loaded = false;  // handle is closed; ibv_lib_loaded() must report false
}

bool ibv_lib_loaded() { return g_loaded; }

bool verbs_category_unavailable(const std::string& category) {
  return category.rfind("verbs", 0) == 0 && !g_loaded;
}

}  // namespace ibvw
}  // namespace acceptance
