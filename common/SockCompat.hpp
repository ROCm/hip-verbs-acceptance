/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// SockCompat — a tiny POSIX-socket shim. The harness needs only a handful of
// TCP/IPv4 calls; wrapping them here keeps call sites small and portable and
// means <sys/socket.h> is included in exactly one place (SockCompat.cpp).
//
// Only the calls the harness needs are wrapped; everything is blocking unless a
// wrapper says otherwise. Errors propagate via errno exactly like the syscalls.

#include <cstddef>

// A small top-level utility namespace, separate from the harness's `acceptance`
// namespace.
namespace sockcompat {

// TCP/IPv4 socket. Returns fd or -1 (errno set).
int tcp_socket();

void close_fd(int fd);

void set_reuseaddr(int fd);
// Set/clear O_NONBLOCK. Returns 0 / -1 (errno set).
int set_nonblocking(int fd, bool on);
void set_tcp_nodelay(int fd);

// Bind to host:port. host may be a dotted-quad or resolvable hostname;
// empty/unresolvable -> INADDR_ANY. -1 on error.
int bind_host_port(int fd, const char* host, int port);
int listen_fd(int fd, int backlog);
// accept() one connection; returns the connected fd or -1.
int accept_one(int fd);

// connect() to host:port. host may be a dotted-quad or a resolvable hostname
// (resolved via getaddrinfo, IPv4). 0 on success, -1 with errno (EINPROGRESS
// for a non-blocking connect in progress; EINVAL if host won't resolve).
int connect_host_port(int fd, const char* host, int port);

// getsockname() port (host byte order), or 0 on error.
int local_port(int fd);

// SO_ERROR for a completed non-blocking connect. Returns the error value via
// *out (0 == success); function returns 0 on success, -1 on getsockopt error.
int get_so_error(int fd, int* out);

// select() for readability/writability on a single fd, with a timeout in
// milliseconds. Returns >0 ready, 0 timeout, -1 error (EINTR possible).
int wait_readable(int fd, long timeout_ms);
int wait_writable(int fd, long timeout_ms);

// Full send/recv with MSG_NOSIGNAL on send. Return bytes transferred or -1.
// These are single send()/recv() calls (callers loop for partial transfer).
long send_some(int fd, const void* buf, size_t len);
long recv_some(int fd, void* buf, size_t len);

// Transfer exactly `len` bytes, looping over partial transfers and retrying
// EINTR. False on a hard error, with errno set, or on a peer that closed before
// `len` bytes moved, with errno untouched: clear errno first to tell them apart.
// MSG_NOSIGNAL on the send path means a dead peer surfaces here rather than as a
// SIGPIPE that kills the writer.
bool send_all(int fd, const void* buf, size_t len);
bool recv_all(int fd, void* buf, size_t len);

}  // namespace sockcompat
