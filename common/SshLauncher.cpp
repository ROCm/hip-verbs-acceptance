/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/SshLauncher.hpp"

#include "common/PosixUtil.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace acceptance {

namespace {

// waitpid_eintr / monotonic_ms live in common/PosixUtil.hpp (shared).
using namespace posix_util;

void set_nonblocking(int fd) {
  const int fl = ::fcntl(fd, F_GETFL, 0);
  if (fl >= 0) ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

// One in-flight ssh child plus its capture pipes.
struct Child {
  pid_t pid = -1;
  int   out_fd = -1;  // read end of child stdout
  int   err_fd = -1;  // read end of child stderr
  bool  reaped = false;
  int   status = 0;
};

// Drain whatever is readable from `fd` into `sink`. Returns false when EOF is
// hit (fd closed) so the caller can stop polling it.
bool drain_fd(int fd, std::string& sink) {
  char buf[4096];
  for (;;) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n > 0) {
      sink.append(buf, static_cast<size_t>(n));
      continue;  // keep draining until EAGAIN/EOF
    }
    if (n == 0) return false;  // EOF
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return true;  // no more for now
    return false;  // read error: treat like EOF
  }
}

}  // namespace

std::string shell_single_quote(const std::string& v) {
  // Wrap in single quotes; replace each embedded ' with '\'' (close, escaped
  // quote, reopen). This is the canonical POSIX-shell single-quote escape and
  // prevents the remote shell from expanding globs, $, spaces, etc.
  std::string out;
  out.reserve(v.size() + 2);
  out.push_back('\'');
  for (const char c : v) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

std::string build_remote_command(const RemoteCommandParams& p, int rank, int size,
                                 const std::string& remote_binary,
                                 const std::string& rendezvous,
                                 const std::string& vendor) {
  // Every interpolated value is single-quoted: these are assembled into one string
  // run by the REMOTE shell via ssh, so an unquoted path/host/vendor containing a
  // space or shell metacharacter would break parsing or allow injection. rank and
  // size are integers we format ourselves, so they need no quoting.
  std::string cmd = shell_single_quote(remote_binary);
  cmd += " --worker";
  cmd += " --rank " + std::to_string(rank);
  cmd += " --size " + std::to_string(size);
  cmd += " --rendezvous " + shell_single_quote(rendezvous);
  cmd += " --results-out -";
  cmd += " --vendor " + shell_single_quote(vendor);
  // Policy dir: resolve to an ABSOLUTE path (the remote worker runs in the ssh
  // home dir, not the launcher's cwd, so a relative --policy would not resolve).
  // On a shared FS the absolute path is identical on every node (same-path
  // assumption). Single-quote in case the path contains spaces.
  if (!p.policy.empty()) {
    std::error_code ec;
    const std::string abs = std::filesystem::absolute(p.policy, ec).string();
    cmd += " --policy " + shell_single_quote(ec ? p.policy : abs);
  }
  // --filter / --filter-category may carry globs — single-quote them.
  if (!p.filter.empty())          cmd += " --filter "          + shell_single_quote(p.filter);
  if (!p.filter_class.empty())    cmd += " --filter-class "    + shell_single_quote(p.filter_class);
  if (!p.filter_category.empty()) cmd += " --filter-category " + shell_single_quote(p.filter_category);
  if (p.hide_na)                  cmd += " --hide-na";
  if (!p.kind.empty())            cmd += " --kind "            + shell_single_quote(p.kind);
  if (p.isolate_explicit)         cmd += " --isolate "         + shell_single_quote(p.isolate);
  if (!p.gid.empty())             cmd += " --gid "             + shell_single_quote(p.gid);
  if (!p.device.empty())          cmd += " --device "          + shell_single_quote(p.device);
  // Worker self-destruct deadline: armed only when > 0, so a 0 (the
  // struct default) omits the flag — keeps the command byte-identical to
  // before this feature existed when the caller never sets it.
  if (p.deadline_sec > 0)
    cmd += " --deadline-sec " + std::to_string(p.deadline_sec);
  return cmd;
}

std::vector<std::string> split_nodes(const std::string& csv) {
  std::vector<std::string> out;
  std::string cur;
  std::stringstream ss(csv);
  while (std::getline(ss, cur, ',')) {
    // trim whitespace
    const auto b = cur.find_first_not_of(" \t");
    const auto e = cur.find_last_not_of(" \t");
    if (b == std::string::npos) continue;
    out.push_back(cur.substr(b, e - b + 1));
  }
  return out;
}

std::vector<std::string> build_ssh_argv(const SshRank& r,
                                        const SshLaunchOptions& opts) {
  // ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no
  //     -o ConnectTimeout=<n> [-l <user>] <node> "<remote_command>"
  //
  //  -n               : ssh must NOT read the launcher's stdin (fan-out safe).
  //  BatchMode=yes    : never prompt (password/passphrase) — fail fast instead.
  //  StrictHostKeyChecking=no : don't block on unknown host keys in automation.
  //  ConnectTimeout   : bound the TCP/handshake so an unreachable node fails
  //                     quickly rather than hanging.
  // The remote command is ONE argv element; ssh hands it to the remote shell.
  std::vector<std::string> argv;
  argv.push_back("ssh");
  argv.push_back("-n");
  argv.push_back("-o");
  argv.push_back("BatchMode=yes");
  argv.push_back("-o");
  argv.push_back("StrictHostKeyChecking=no");
  argv.push_back("-o");
  argv.push_back("ConnectTimeout=" + std::to_string(opts.connect_timeout));
  if (!opts.ssh_user.empty()) {
    argv.push_back("-l");
    argv.push_back(opts.ssh_user);
  }
  argv.push_back(r.node);
  argv.push_back(r.remote_command);
  return argv;
}

std::vector<SshChildResult> ssh_launch(const std::vector<SshRank>& ranks,
                                       const SshLaunchOptions& opts) {
  const size_t n = ranks.size();
  std::vector<SshChildResult> results(n);
  std::vector<Child> children(n);
  for (size_t i = 0; i < n; ++i) {
    results[i].rank = ranks[i].rank;
    results[i].node = ranks[i].node;
  }

  // --- spawn all ssh children ---
  for (size_t i = 0; i < n; ++i) {
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    // O_CLOEXEC so a sibling child's pipe fds don't leak into a later child
    // (and close automatically across exec). The child's own ends are made to
    // survive via dup2 onto std fds (dup2 clears CLOEXEC on the target).
    if (::pipe2(out_pipe, O_CLOEXEC) != 0 || ::pipe2(err_pipe, O_CLOEXEC) != 0) {
      // Pipe creation failed: mark not-launched and close any opened ends.
      // pipe2() leaves its array untouched on failure, so only the first (out)
      // pipe can be open if the second (err) failed; close both guarded so the
      // cleanup stays correct regardless of which pipe2() failed.
      if (out_pipe[0] >= 0) { ::close(out_pipe[0]); ::close(out_pipe[1]); }
      if (err_pipe[0] >= 0) { ::close(err_pipe[0]); ::close(err_pipe[1]); }
      results[i].launched = false;
      results[i].stderr_data = std::string("pipe() failed: ") +
                               std::strerror(errno);
      continue;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
      ::close(out_pipe[0]); ::close(out_pipe[1]);
      ::close(err_pipe[0]); ::close(err_pipe[1]);
      results[i].launched = false;
      results[i].stderr_data = std::string("fork() failed: ") +
                               std::strerror(errno);
      continue;
    }

    if (pid == 0) {
      // ---- child: wire stdout/stderr to the pipes, exec ssh ----
      ::dup2(out_pipe[1], STDOUT_FILENO);
      ::dup2(err_pipe[1], STDERR_FILENO);
      // Close all original pipe fds (both ends) after duping.
      ::close(out_pipe[0]); ::close(out_pipe[1]);
      ::close(err_pipe[0]); ::close(err_pipe[1]);
      // ssh -n already disconnects stdin, but redirect from /dev/null too so the
      // child can never touch the launcher's stdin.
      const int devnull = ::open("/dev/null", O_RDONLY);
      if (devnull >= 0) { ::dup2(devnull, STDIN_FILENO); ::close(devnull); }

      const std::vector<std::string> argv = build_ssh_argv(ranks[i], opts);
      std::vector<char*> cargv;
      cargv.reserve(argv.size() + 1);
      // execvp does not modify argv, but the signature wants char* const*.
      for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
      cargv.push_back(nullptr);
      ::execvp("ssh", cargv.data());
      // exec failed — emit a diagnostic on the (redirected) stderr and exit 127.
      const char* msg = "hva: failed to exec ssh\n";
      (void)!::write(STDERR_FILENO, msg, std::strlen(msg));
      ::_exit(127);
    }

    // ---- parent: keep read ends, close write ends ----
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);
    set_nonblocking(out_pipe[0]);
    set_nonblocking(err_pipe[0]);
    children[i].pid = pid;
    children[i].out_fd = out_pipe[0];
    children[i].err_fd = err_pipe[0];
    results[i].launched = true;
  }

  // --- bounded concurrent wait: drain pipes + reap, under one deadline ---
  const long t0_ms = monotonic_ms();
  bool deadline_hit = false;

  auto any_unreaped = [&]() {
    for (size_t i = 0; i < n; ++i) {
      if (results[i].launched && !children[i].reaped) return true;
    }
    return false;
  };

  while (any_unreaped()) {
    // Build the read set from still-open fds.
    fd_set rfds;
    FD_ZERO(&rfds);
    int maxfd = -1;
    for (size_t i = 0; i < n; ++i) {
      if (!results[i].launched) continue;
      if (children[i].out_fd >= 0) {
        FD_SET(children[i].out_fd, &rfds);
        if (children[i].out_fd > maxfd) maxfd = children[i].out_fd;
      }
      if (children[i].err_fd >= 0) {
        FD_SET(children[i].err_fd, &rfds);
        if (children[i].err_fd > maxfd) maxfd = children[i].err_fd;
      }
    }

    if (maxfd >= 0) {
      struct timeval tv { 0, 200 * 1000 };  // 200ms cadence
      const int sr = ::select(maxfd + 1, &rfds, nullptr, nullptr, &tv);
      if (sr > 0) {
        for (size_t i = 0; i < n; ++i) {
          if (!results[i].launched) continue;
          if (children[i].out_fd >= 0 && FD_ISSET(children[i].out_fd, &rfds)) {
            if (!drain_fd(children[i].out_fd, results[i].stdout_data)) {
              ::close(children[i].out_fd);
              children[i].out_fd = -1;
            }
          }
          if (children[i].err_fd >= 0 && FD_ISSET(children[i].err_fd, &rfds)) {
            if (!drain_fd(children[i].err_fd, results[i].stderr_data)) {
              ::close(children[i].err_fd);
              children[i].err_fd = -1;
            }
          }
        }
      } else if (sr < 0 && errno != EINTR) {
        // Unexpected select error: stop polling fds (still reap below).
        for (size_t i = 0; i < n; ++i) {
          if (children[i].out_fd >= 0) { ::close(children[i].out_fd); children[i].out_fd = -1; }
          if (children[i].err_fd >= 0) { ::close(children[i].err_fd); children[i].err_fd = -1; }
        }
      }
    } else {
      // No open fds but children still running: short nap to avoid a busy spin.
      struct timespec nap { 0, 100 * 1000 * 1000L };  // 100ms
      ::nanosleep(&nap, nullptr);
    }

    // Non-blocking reap of any finished children.
    for (size_t i = 0; i < n; ++i) {
      if (!results[i].launched || children[i].reaped) continue;
      int st = 0;
      const pid_t r = waitpid_eintr(children[i].pid, &st, WNOHANG);
      if (r == children[i].pid) {
        children[i].reaped = true;
        children[i].status = st;
      } else if (r < 0) {
        // Already gone / no such child: mark reaped to avoid spinning.
        children[i].reaped = true;
        children[i].status = -1;
      }
    }

    if ((monotonic_ms() - t0_ms) / 1000 >= opts.overall_timeout_sec) {
      deadline_hit = true;
      break;
    }
  }

  // --- deadline overrun: SIGTERM -> grace -> SIGKILL, then reap all ---
  if (deadline_hit) {
    for (size_t i = 0; i < n; ++i) {
      if (results[i].launched && !children[i].reaped) {
        ::kill(children[i].pid, SIGTERM);
        results[i].timed_out = true;
      }
    }
    // Grace window for SIGTERM to take effect.
    const long grace_ms = 2000;
    long g = 0;
    while (g < grace_ms && any_unreaped()) {
      for (size_t i = 0; i < n; ++i) {
        if (!results[i].launched || children[i].reaped) continue;
        int st = 0;
        const pid_t r = waitpid_eintr(children[i].pid, &st, WNOHANG);
        if (r == children[i].pid) { children[i].reaped = true; children[i].status = st; }
        else if (r < 0) { children[i].reaped = true; children[i].status = -1; }
      }
      struct timespec nap { 0, 50 * 1000 * 1000L };  // 50ms
      ::nanosleep(&nap, nullptr);
      g += 50;
    }
    // Escalate to SIGKILL and blocking-reap any survivors.
    for (size_t i = 0; i < n; ++i) {
      if (results[i].launched && !children[i].reaped) {
        ::kill(children[i].pid, SIGKILL);
        int st = 0;
        waitpid_eintr(children[i].pid, &st, 0);
        children[i].reaped = true;
        children[i].status = st;
      }
    }
  }

  // --- close remaining fds + drain final bytes, fill exit info ---
  for (size_t i = 0; i < n; ++i) {
    if (!results[i].launched) continue;
    // Final non-blocking drain of any bytes buffered before close.
    if (children[i].out_fd >= 0) {
      drain_fd(children[i].out_fd, results[i].stdout_data);
      ::close(children[i].out_fd);
      children[i].out_fd = -1;
    }
    if (children[i].err_fd >= 0) {
      drain_fd(children[i].err_fd, results[i].stderr_data);
      ::close(children[i].err_fd);
      children[i].err_fd = -1;
    }
    const int st = children[i].status;
    if (st == -1) {
      results[i].exited = false;
    } else if (WIFEXITED(st)) {
      results[i].exited = true;
      results[i].exit_code = WEXITSTATUS(st);
    } else if (WIFSIGNALED(st)) {
      results[i].exited = false;
      results[i].term_signal = WTERMSIG(st);
    }
  }

  return results;
}

}  // namespace acceptance
