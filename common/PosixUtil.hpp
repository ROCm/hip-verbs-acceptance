/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once
//
// Small POSIX leaf helpers shared by the harness's process-control paths: the
// Runner watchdog, the SshLauncher fan-out, the local-ranks launcher, and the
// SocketBootstrap deadlines. Header-only (no extra TU); each function is a thin,
// dependency-free wrapper that was previously copy-pasted into several files.
//
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <ctime>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace acceptance {
namespace posix_util {

// EINTR-safe waitpid. `flags` may include WNOHANG. Returns the waitpid() value
// (pid, 0 for WNOHANG-no-change, or -1 with errno set).
inline pid_t waitpid_eintr(pid_t pid, int* status, int flags) {
  for (;;) {
    const pid_t r = ::waitpid(pid, status, flags);
    if (r < 0 && errno == EINTR) continue;
    return r;
  }
}

// Sleep `ms` milliseconds, resuming across EINTR so a watchdog/backoff cadence
// is not cut short by signal delivery.
inline void sleep_ms(long ms) {
  struct timespec req {};
  req.tv_sec = ms / 1000;
  req.tv_nsec = (ms % 1000) * 1000000L;
  struct timespec rem {};
  while (::nanosleep(&req, &rem) != 0 && errno == EINTR) {
    req = rem;
  }
}

// Returned when no wait status could be obtained because the child was already
// reaped; not a value waitpid() can produce, so it never reads as a clean exit.
constexpr int kWaitStatusUnavailable = -1;

// Reap `pid`, polling for up to `grace_ms` and then SIGKILLing it, so a wedged
// child cannot block the caller indefinitely. Returns the wait status.
inline int reap_child_bounded(pid_t pid, long grace_ms) {
  int status = 0;
  for (long waited_ms = 0; waited_ms < grace_ms; ++waited_ms) {
    const pid_t reaped = waitpid_eintr(pid, &status, WNOHANG);
    if (reaped == pid) return status;
    if (reaped < 0) return kWaitStatusUnavailable;
    sleep_ms(1);
  }
  (void)::kill(pid, SIGKILL);
  (void)waitpid_eintr(pid, &status, 0);
  return status;
}

// Monotonic clock in milliseconds, for deadlines/elapsed (never wall-clock, so
// it is immune to clock steps). Seconds are just `monotonic_ms() / 1000`.
inline long monotonic_ms() {
  struct timespec ts {};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<long>(ts.tv_sec) * 1000L + ts.tv_nsec / 1000000L;
}

// A wait status that means the child process ran to completion and succeeded.
inline bool exited_clean(int status) {
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// Reap `count` child processes against a single `grace_ms` deadline, SIGKILLing
// the survivors. Entries are zeroed as they are reaped. Returns how many exited
// with status 0.
inline int reap_children_bounded(pid_t* pids, size_t count, long grace_ms) {
  size_t outstanding = count;
  int clean = 0;
  const long deadline = monotonic_ms() + grace_ms;

  while (outstanding > 0 && monotonic_ms() < deadline) {
    for (size_t i = 0; i < count; ++i) {
      if (pids[i] == 0) continue;

      int status = 0;
      const pid_t reaped = waitpid_eintr(pids[i], &status, WNOHANG);
      if (reaped == 0) continue;
      if (reaped == pids[i] && exited_clean(status)) ++clean;

      pids[i] = 0;
      --outstanding;
    }
    if (outstanding > 0) sleep_ms(1);
  }

  // Blocking now, because these are being killed.
  for (size_t i = 0; i < count; ++i) {
    if (pids[i] == 0) continue;

    (void)::kill(pids[i], SIGKILL);
    int status = 0;
    const pid_t reaped = waitpid_eintr(pids[i], &status, 0);
    if (reaped == pids[i] && exited_clean(status)) ++clean;

    pids[i] = 0;
  }
  return clean;
}

}  // namespace posix_util
}  // namespace acceptance
