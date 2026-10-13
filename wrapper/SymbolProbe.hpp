/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// SymbolProbe — host-only dlopen/dlsym helpers for Symbol_Version_Detection.
// These resolve OPTIONAL runtime symbols (HSA dmabuf export, cuMem family, the
// ionic direct-verbs library) without linking them, so the binary loads and
// reports cleanly on a node that lacks them. Gated on ACCEPT_HAVE_HIP so the
// header is a no-op in non-HIP builds.

#ifdef ACCEPT_HAVE_HIP

#include <dlfcn.h>

#include <string>

namespace acceptance {

// True if `lib` can be dlopen'd (lazy, local). Leaks the handle deliberately:
// the probe is one-shot at test time and keeping it open avoids unloading a
// library the HIP/HSA runtime may already depend on.
[[maybe_unused]] static bool hip_lib_present(const char* lib) {
  void* h = ::dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
  return h != nullptr;
}

// True if `symbol` resolves in `lib` (dlopen then dlsym). Falls back to the
// global namespace (RTLD_DEFAULT) when the library itself is not present, so a
// symbol already pulled in by the linked HIP/HSA runtime is still detected.
[[maybe_unused]] static bool hip_symbol_present(const char* lib,
                                                const char* symbol) {
  if (void* h = ::dlopen(lib, RTLD_LAZY | RTLD_LOCAL)) {
    if (::dlsym(h, symbol) != nullptr) return true;
  }
  return ::dlsym(RTLD_DEFAULT, symbol) != nullptr;
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP
