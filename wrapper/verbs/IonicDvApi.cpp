/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#ifdef ACCEPT_HAVE_IBVERBS

// dlvsym is a GNU extension declared by <dlfcn.h> only under _GNU_SOURCE. g++
// and ROCm clang++ define it implicitly for C++, but don't rely on that -- set
// it explicitly before any header so a strict -std=c++20 / libc++ build still
// sees the declaration.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "wrapper/verbs/IonicDvApi.hpp"

#include <dlfcn.h>

#include <initializer_list>
#include <mutex>

namespace acceptance {
namespace ionicdv {

namespace {

constexpr char kIonicVersion[] = "IONIC_1.0";  // RCCL's IONIC_VERSION node

std::once_flag      g_once;
void*               g_handle = nullptr;
pd_set_udma_mask_fn g_set_udma = nullptr;
qp_set_gda_fn       g_set_gda  = nullptr;

void load_once() {
  // Same soname order as the presence probe in common/Capability.cpp.
  // RTLD_NODELETE: never unmap the driver libionic registers into libibverbs.
  for (const char* name : {"libionic.so.1", "libionic.so"}) {
    g_handle = ::dlopen(name, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (g_handle != nullptr) break;
  }
  if (g_handle == nullptr) return;

  g_set_udma = reinterpret_cast<pd_set_udma_mask_fn>(
      ::dlvsym(g_handle, "ionic_dv_pd_set_udma_mask", kIonicVersion));
  g_set_gda = reinterpret_cast<qp_set_gda_fn>(
      ::dlvsym(g_handle, "ionic_dv_qp_set_gda", kIonicVersion));

  if (g_set_udma == nullptr || g_set_gda == nullptr) {
    g_set_udma = nullptr;  // partial load -> unavailable
    g_set_gda = nullptr;
  }
}

}  // namespace

bool init() {
  std::call_once(g_once, load_once);
  return available();
}

bool available() { return g_set_udma != nullptr && g_set_gda != nullptr; }

pd_set_udma_mask_fn pd_set_udma_mask() { return g_set_udma; }
qp_set_gda_fn       qp_set_gda()       { return g_set_gda; }

}  // namespace ionicdv
}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
