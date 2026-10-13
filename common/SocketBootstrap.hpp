/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// SocketBootstrap — net-new TCP rendezvous for size in {1, 2} (ARCHITECTURE.md
// §4). It is the runtime out-of-band (OOB) primitive the data plane uses to
// rendezvous and swap small metadata; it is NOT a cross-rank pass/fail consensus
// (§4 non-goal). Each rank reports its own honest local outcome; the Reporter
// reconciles (§7.3).
//
// Scope (deliberate, from review): exactly size 1 or 2.
//   * size == 1 — behaves like LoopbackBootstrap: no socket; barrier is a no-op,
//     allgather/exchange are local copies.
//   * size == 2 — rank 0 binds+listens+accepts ONE peer; rank 1 connects. ONE
//     persistent TCP connection is reused for every barrier/exchange/allgather
//     (not a fresh port per op). Both accept and connect are BOUNDED by a timeout
//     (no infinite retry, unlike qp_link_probe.c); on timeout/failure the
//     constructor throws std::runtime_error.
//
// Limitations:
//   * size > 2 is unsupported (throws in the ctor).
//   * exchange()/allgather() use a write-then-read ordering, correct only for the
//     small payloads this harness uses (tiny QP-metadata / ints). It is NOT a
//     deadlock-free full-duplex transfer for large buffers; that is out of scope
//     for Phase 1 (see ARCHITECTURE §4 / §13).

#include <cstddef>
#include <string>

#include "common/Bootstrap.hpp"

namespace acceptance {

class SocketBootstrap : public Bootstrap {
 public:
  // Construct + rendezvous. For size 2 this performs the bind/accept (rank 0) or
  // connect (rank 1) with a bounded timeout, leaving one connected socket live
  // for the object's lifetime. Throws std::runtime_error on bad args
  // (size > 2, rank out of range), or on rendezvous timeout/failure.
  SocketBootstrap(int rank, int size, const std::string& rendezvous_host,
                  int rendezvous_port, int timeout_sec = 30);

  ~SocketBootstrap() override;

  SocketBootstrap(const SocketBootstrap&) = delete;
  SocketBootstrap& operator=(const SocketBootstrap&) = delete;

  int  rank() const override { return rank_; }
  int  size() const override { return size_; }

  // Rendezvous: exchange a 1-byte token in both directions so both ranks
  // synchronize. No-op at size 1.
  void barrier() override;

  // allgather: place own `in` at out[rank*len], peer's at out[(1-rank)*len].
  // size 1 -> local copy. size 2 -> one swap with the peer.
  void allgather(const void* in, void* out, size_t len) override;

  // exchange: swap `len` bytes with `peer` (the other rank for size 2). size 1
  // -> local copy (only self is reachable).
  void exchange(int peer, const void* s, void* r, size_t len) override;

 private:
  int rank_ = 0;
  int size_ = 1;
  int fd_   = -1;   // the one persistent connected socket (size 2); -1 at size 1

  // Rendezvous helpers (size 2). Each throws std::runtime_error on failure.
  void listen_and_accept(const std::string& host, int port, int timeout_sec);
  void connect_to(const std::string& host, int port, int timeout_sec);
};

}  // namespace acceptance
