/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/SocketBootstrap.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <time.h>

#include "common/PosixUtil.hpp"
#include "common/SockCompat.hpp"

namespace acceptance {

namespace {

namespace sc = ::sockcompat;

// monotonic_ms / sleep_ms live in common/PosixUtil.hpp (shared).
using namespace posix_util;

std::string sys_err(const std::string& what) {
  return what + ": " + std::strerror(errno);
}

// Full write, throwing on hard error or peer close mid-write. errno is cleared
// first because sc::send_all leaves it untouched when the peer closed.
void write_all(int fd, const void* buf, size_t len) {
  errno = 0;
  if (sc::send_all(fd, buf, len)) return;
  if (errno != 0) throw std::runtime_error(sys_err("SocketBootstrap: send"));
  throw std::runtime_error("SocketBootstrap: send returned 0 (peer closed)");
}

// Full read, throwing on hard error or premature EOF.
void read_all(int fd, void* buf, size_t len) {
  errno = 0;
  if (sc::recv_all(fd, buf, len)) return;
  if (errno != 0) throw std::runtime_error(sys_err("SocketBootstrap: recv"));
  throw std::runtime_error(
      "SocketBootstrap: recv returned 0 (peer closed early)");
}

}  // namespace

SocketBootstrap::SocketBootstrap(int rank, int size,
                                 const std::string& rendezvous_host,
                                 int rendezvous_port, int timeout_sec)
    : rank_(rank), size_(size) {
  if (size < 1 || size > 2) {
    throw std::runtime_error(
        "SocketBootstrap: size must be 1 or 2 (got " + std::to_string(size) +
        ")");
  }
  if (rank < 0 || rank >= size) {
    throw std::runtime_error("SocketBootstrap: rank out of range (rank=" +
                             std::to_string(rank) + ", size=" +
                             std::to_string(size) + ")");
  }
  if (size == 1) return;  // loopback; no socket

  if (rank == 0) {
    listen_and_accept(rendezvous_host, rendezvous_port, timeout_sec);
  } else {
    connect_to(rendezvous_host, rendezvous_port, timeout_sec);
  }
}

SocketBootstrap::~SocketBootstrap() {
  if (fd_ >= 0) sc::close_fd(fd_);
}

// Rank 0: bind + listen, then accept exactly one peer within timeout_sec.
void SocketBootstrap::listen_and_accept(const std::string& host, int port,
                                        int timeout_sec) {
  (void)host;  // The listener binds ALL interfaces (0.0.0.0), NOT the rendezvous
               // host: a node resolving its own FQDN often hits 127.0.1.1
               // (/etc/hosts on Ubuntu), which would bind loopback and be
               // unreachable from the peer. The rendezvous host is meaningful
               // only to the connector (rank 1).
  const int lfd = sc::tcp_socket();
  if (lfd < 0) throw std::runtime_error(sys_err("SocketBootstrap: socket"));
  sc::set_reuseaddr(lfd);

  if (sc::bind_host_port(lfd, "", port) < 0) {  // "" => INADDR_ANY
    const std::string e = sys_err("SocketBootstrap: bind");
    sc::close_fd(lfd);
    throw std::runtime_error(e);
  }
  if (sc::listen_fd(lfd, 1) < 0) {
    const std::string e = sys_err("SocketBootstrap: listen");
    sc::close_fd(lfd);
    throw std::runtime_error(e);
  }

  // Bounded accept: non-blocking listener + select() on the deadline.
  sc::set_nonblocking(lfd, true);
  const long deadline = monotonic_ms() + static_cast<long>(timeout_sec) * 1000L;
  for (;;) {
    const long remain = deadline - monotonic_ms();
    if (remain <= 0) {
      sc::close_fd(lfd);
      throw std::runtime_error(
          "SocketBootstrap: accept timed out after " +
          std::to_string(timeout_sec) + "s (peer never connected)");
    }
    const int s = sc::wait_readable(lfd, remain);
    if (s < 0) {
      if (errno == EINTR) continue;
      const std::string e = sys_err("SocketBootstrap: select(accept)");
      sc::close_fd(lfd);
      throw std::runtime_error(e);
    }
    if (s == 0) continue;  // re-check deadline

    const int cfd = sc::accept_one(lfd);
    if (cfd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      const std::string e = sys_err("SocketBootstrap: accept");
      sc::close_fd(lfd);
      throw std::runtime_error(e);
    }
    sc::close_fd(lfd);            // one peer only; stop listening
    sc::set_nonblocking(cfd, false);  // blocking for the data primitives
    sc::set_tcp_nodelay(cfd);
    fd_ = cfd;
    return;
  }
}

// Rank 1: connect to host:port within timeout_sec (non-blocking connect + poll).
void SocketBootstrap::connect_to(const std::string& host, int port,
                                 int timeout_sec) {
  const std::string h = host.empty() ? "127.0.0.1" : host;
  const long deadline = monotonic_ms() + static_cast<long>(timeout_sec) * 1000L;

  // Retry the whole connect attempt until the deadline: the listener may not be
  // up yet (the launcher forks both ranks concurrently). Each attempt is itself
  // bounded by the remaining time — never an infinite retry.
  for (;;) {
    const long remain = deadline - monotonic_ms();
    if (remain <= 0) {
      throw std::runtime_error(
          "SocketBootstrap: connect timed out after " +
          std::to_string(timeout_sec) + "s (no listener at " + h + ":" +
          std::to_string(port) + ")");
    }

    const int fd = sc::tcp_socket();
    if (fd < 0) throw std::runtime_error(sys_err("SocketBootstrap: socket"));
    sc::set_nonblocking(fd, true);

    const int rc = sc::connect_host_port(fd, h.c_str(), port);
    if (rc == 0) {
      sc::set_nonblocking(fd, false);
      sc::set_tcp_nodelay(fd);
      fd_ = fd;
      return;
    }
    if (errno != EINPROGRESS) {
      // No listener yet (ECONNREFUSED) or transient/EINVAL: back off, retry.
      sc::close_fd(fd);
      sleep_ms(20);
      continue;
    }

    const long w_remain = deadline - monotonic_ms();
    if (w_remain <= 0) { sc::close_fd(fd); continue; }
    const int s = sc::wait_writable(fd, w_remain);
    if (s < 0) {
      if (errno == EINTR) { sc::close_fd(fd); continue; }
      const std::string e = sys_err("SocketBootstrap: select(connect)");
      sc::close_fd(fd);
      throw std::runtime_error(e);
    }
    if (s == 0) { sc::close_fd(fd); continue; }  // timed out this round

    int so_err = 0;
    if (sc::get_so_error(fd, &so_err) < 0) {
      const std::string e = sys_err("SocketBootstrap: getsockopt(SO_ERROR)");
      sc::close_fd(fd);
      throw std::runtime_error(e);
    }
    if (so_err != 0) {
      sc::close_fd(fd);
      sleep_ms(20);
      continue;
    }
    sc::set_nonblocking(fd, false);
    sc::set_tcp_nodelay(fd);
    fd_ = fd;
    return;
  }
}

void SocketBootstrap::barrier() {
  if (size_ == 1) return;
  // Both ranks send a 1-byte token then read the peer's, so neither proceeds
  // until both have arrived. One byte cannot deadlock write-then-read (it fits
  // the kernel send buffer trivially).
  const char tok = 'B';
  char peer = 0;
  write_all(fd_, &tok, 1);
  read_all(fd_, &peer, 1);
}

void SocketBootstrap::exchange(int peer, const void* s, void* r, size_t len) {
  if (size_ == 1) {
    (void)peer;  // only self is reachable
    if (len) std::memcpy(r, s, len);
    return;
  }
  // size 2: the peer is always the other rank. Write-then-read is correct for
  // the small payloads this harness uses (see header limitation note).
  if (len) write_all(fd_, s, len);
  if (len) read_all(fd_, r, len);
}

void SocketBootstrap::allgather(const void* in, void* out, size_t len) {
  if (size_ == 1) {
    if (len) std::memcpy(out, in, len);
    return;
  }
  char* o = static_cast<char*>(out);
  if (len) std::memcpy(o + static_cast<size_t>(rank_) * len, in, len);
  const int peer = 1 - rank_;
  exchange(peer, in, o + static_cast<size_t>(peer) * len, len);
}

}  // namespace acceptance
