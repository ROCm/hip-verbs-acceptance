/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// SockCompat.cpp — TCP/IPv4 shim; sole TU that includes <sys/socket.h>.

#include "common/SockCompat.hpp"

#include <cstring>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cerrno>
#include <cstdint>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace sockcompat {

int tcp_socket() { return ::socket(AF_INET, SOCK_STREAM, 0); }

void close_fd(int fd) { ::close(fd); }

void set_reuseaddr(int fd) {
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
}

int set_nonblocking(int fd, bool on) {
  int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) return -1;
  if (on) flags |= O_NONBLOCK;
  else    flags &= ~O_NONBLOCK;
  return ::fcntl(fd, F_SETFL, flags);
}

void set_tcp_nodelay(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static bool fill_addr(struct sockaddr_in* a, const char* host, int port,
                      bool any_on_fail) {
  std::memset(a, 0, sizeof(*a));
  a->sin_family = AF_INET;
  a->sin_port = htons(static_cast<uint16_t>(port));
  if (host != nullptr && host[0] != '\0') {
    // Fast path: numeric dotted-quad literal.
    if (::inet_pton(AF_INET, host, &a->sin_addr) == 1) return true;
    // Resolve hostname when not a dotted-quad literal.
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    const int rc = ::getaddrinfo(host, nullptr, &hints, &res);
    if (rc == 0 && res != nullptr) {
      a->sin_addr =
          reinterpret_cast<struct sockaddr_in*>(res->ai_addr)->sin_addr;
      ::freeaddrinfo(res);
      return true;
    }
    if (res != nullptr) ::freeaddrinfo(res);
  }
  // INADDR_ANY only when host is empty; non-empty unresolved host is an error.
  if (any_on_fail && (host == nullptr || host[0] == '\0')) {
    a->sin_addr.s_addr = htonl(INADDR_ANY);
    return true;
  }
  return false;
}

int bind_host_port(int fd, const char* host, int port) {
  struct sockaddr_in a;
  if (!fill_addr(&a, host, port, /*any_on_fail=*/true)) {
    errno = EADDRNOTAVAIL;  // non-empty bind host that did not resolve
    return -1;
  }
  return ::bind(fd, reinterpret_cast<struct sockaddr*>(&a), sizeof(a));
}

int listen_fd(int fd, int backlog) { return ::listen(fd, backlog); }

int accept_one(int fd) { return ::accept(fd, nullptr, nullptr); }

int connect_host_port(int fd, const char* host, int port) {
  struct sockaddr_in a;
  if (!fill_addr(&a, host, port, /*any_on_fail=*/false)) {
    errno = EINVAL;
    return -1;
  }
  return ::connect(fd, reinterpret_cast<struct sockaddr*>(&a), sizeof(a));
}

int local_port(int fd) {
  struct sockaddr_in a;
  socklen_t len = sizeof(a);
  if (::getsockname(fd, reinterpret_cast<struct sockaddr*>(&a), &len) != 0) {
    return 0;
  }
  return ntohs(a.sin_port);
}

int get_so_error(int fd, int* out) {
  int so_err = 0;
  socklen_t len = sizeof(so_err);
  if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) < 0) return -1;
  if (out) *out = so_err;
  return 0;
}

static int wait_one(int fd, long timeout_ms, bool want_write) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd, &fds);
  struct timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  if (want_write) return ::select(fd + 1, nullptr, &fds, nullptr, &tv);
  return ::select(fd + 1, &fds, nullptr, nullptr, &tv);
}

int wait_readable(int fd, long timeout_ms) {
  return wait_one(fd, timeout_ms, /*want_write=*/false);
}
int wait_writable(int fd, long timeout_ms) {
  return wait_one(fd, timeout_ms, /*want_write=*/true);
}

long send_some(int fd, const void* buf, size_t len) {
  return ::send(fd, buf, len, MSG_NOSIGNAL);
}
long recv_some(int fd, void* buf, size_t len) {
  return ::recv(fd, buf, len, 0);
}

bool send_all(int fd, const void* buf, size_t len) {
  const char* cursor = static_cast<const char*>(buf);
  while (len > 0) {
    const long sent = send_some(fd, cursor, len);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) return false;
    cursor += sent;
    len -= static_cast<size_t>(sent);
  }
  return true;
}

bool recv_all(int fd, void* buf, size_t len) {
  char* cursor = static_cast<char*>(buf);
  while (len > 0) {
    const long got = recv_some(fd, cursor, len);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    cursor += got;
    len -= static_cast<size_t>(got);
  }
  return true;
}

}  // namespace sockcompat
