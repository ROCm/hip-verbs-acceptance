/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/TestRegistry.hpp"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace acceptance {

namespace {
// Meyers singleton: constructed on first use, so registration from any TU's
// static initializer is safe regardless of init order.
std::vector<TestMeta>& registry() {
  static std::vector<TestMeta> tests;
  return tests;
}
}  // namespace

bool TestRegistry::add(const TestMeta& m) {
  // Duplicate ids shadow policy lookup; keep the first registration.
  for (const auto& existing : registry()) {
    if (existing.id == m.id) {
      std::fprintf(stderr,
                   "hva: WARNING duplicate test id '%s' ignored "
                   "(keeping the first registration)\n",
                   m.id.c_str());
      return false;
    }
  }
  registry().push_back(m);
  return true;
}

const std::vector<TestMeta>& TestRegistry::all() {
  // Stable sort by category so all() always presents tests grouped by suite,
  // regardless of TU link order. Stable preserves id order within a category.
  // std::call_once guarantees the sort runs exactly once even under concurrent
  // calls (the harness is single-threaded today, but this is correct either way).
  static std::once_flag sort_once;
  std::call_once(sort_once, [] {
    std::stable_sort(registry().begin(), registry().end(),
                     [](const TestMeta& a, const TestMeta& b) {
                       return a.category < b.category;
                     });
  });
  return registry();
}

const TestMeta* TestRegistry::find(const std::string& id) {
  for (const auto& m : registry()) {
    if (m.id == id) return &m;
  }
  return nullptr;
}

}  // namespace acceptance
