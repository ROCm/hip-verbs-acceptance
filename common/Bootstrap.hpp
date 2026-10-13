/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Bootstrap — minimal collective primitives a test uses to coordinate across
// ranks. The interface is fixed here in Step 1; a real socket-backed
// implementation arrives in Step 5. LoopbackBootstrap is the size==1 stand-in
// used by EXEC mode and by single-rank tests.

#include <cstring>
#include <stdexcept>

namespace acceptance {

class Bootstrap {
public:
  virtual int  rank() const = 0;
  virtual int  size() const = 0;
  virtual void barrier() = 0;
  // allgather: gather one contiguous `len`-byte block from each rank into `out`
  // (out must hold size()*len bytes). size()<=2 for this harness.
  virtual void allgather(const void* in, void* out, size_t len) = 0;
  // exchange: pairwise send `s`/recv `r`, each `len` bytes, with `peer`.
  virtual void exchange(int peer, const void* s, void* r, size_t len) = 0;
  virtual ~Bootstrap() = default;
};

// Single-rank bootstrap: rank 0 of 1. Collectives degenerate to local copies.
class LoopbackBootstrap : public Bootstrap {
public:
  int  rank() const override { return 0; }
  int  size() const override { return 1; }
  void barrier() override {}
  void allgather(const void* in, void* out, size_t len) override {
    if (len) std::memcpy(out, in, len);
  }
  void exchange(int peer, const void* s, void* r, size_t len) override {
    (void)peer;  // only self is reachable at size 1
    if (len) std::memcpy(r, s, len);
  }
};

}  // namespace acceptance
