/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// SshLauncher — the in-binary ssh/srun multi-node launcher (Phase 1b / Plan
// Step 6).
//
// Given a node list and a per-rank remote command, it fans out one child
// per rank (an `ssh` or an `srun` process, selected by SshLaunchOptions::launcher),
// captures each child's stdout (the per-rank results block, produced
// by the worker's `--results-out -`) and stderr (human/log output) on separate
// pipes, waits for all children under a single overall wall-clock deadline, and
// kills/reaps any that overrun. Nothing hangs: every wait is bounded and every
// child is reaped (no zombies); every pipe end is closed (no fd leaks).
//
// SAME-PATH ASSUMPTION (NFS / shared FS deployments): the remote binary
// and the policy directory must be reachable at the SAME absolute path on every
// node. The launcher defaults the remote binary to its own /proc/self/exe and
// passes the policy dir through unchanged; both are overridable on the CLI.
//
// Scope: size in {1, 2} only (mirrors SocketBootstrap's size<=2 scope). The
// rendezvous is nodes[0]:port — rank 0 runs on nodes[0] (listener binds
// 0.0.0.0), rank 1 on nodes[1] (dials nodes[0]:port). Under --remote-launcher
// srun, rank 0's node is launched via `srun --nodelist=<node>` exactly like any
// other rank — there is no local-node special case.
//
// Deadline handling is identical for both launchers: on overrun, ssh_launch
// SIGTERMs the child, waits a 2s grace period, then SIGKILLs. For ssh this
// kills the local ssh client (the remote worker may linger — worker
// self-cleanup is handled separately). For srun, srun forwards SIGTERM to the remote task step, and
// if srun itself is SIGKILLed, Slurm's slurmd/slurmstepd cleans up the step on
// that node so the task does not leak past the launcher's exit.

#include <string>
#include <utility>
#include <vector>

namespace acceptance {

// Which external tool launches each rank's worker process.
enum class RemoteLauncher { Ssh, Srun };

// Per-rank spec the launcher needs to build one child (ssh or srun).
struct SshRank {
  int         rank = 0;
  std::string node;             // hostname/FQDN (ssh target, or srun --nodelist)
  std::string remote_command;   // the full remote command as ONE shell string
};

// Outcome of one launcher child (ssh or srun).
struct SshChildResult {
  int         rank = 0;
  std::string node;
  bool        launched = false; // fork+exec of the launcher binary succeeded
  bool        exited = false;   // process reaped via normal exit
  bool        timed_out = false;// killed because the overall deadline elapsed
  int         exit_code = -1;   // launcher exit status (ssh: 255 = ssh-level failure)
  int         term_signal = 0;  // nonzero if killed by signal
  std::string stdout_data;      // captured stdout (the per-rank results block)
  std::string stderr_data;      // captured stderr (worker logs / launcher errors)
};

// Options controlling how the remote worker is launched.
struct SshLaunchOptions {
  RemoteLauncher launcher = RemoteLauncher::Ssh;  // --remote-launcher ssh|srun
  std::string ssh_user;          // -l <user> (empty => omit); Ssh only, unused by Srun
  int         connect_timeout = 10;  // ssh -o ConnectTimeout; Ssh only, unused by Srun
  long        overall_timeout_sec = 120;  // overall wall-clock deadline
};

// Run all ranks concurrently. Returns one SshChildResult per input rank (same
// order). Bounded by opts.overall_timeout_sec; overrunning children are
// SIGTERM->SIGKILL'd and reaped. Never throws; never hangs.
std::vector<SshChildResult> ssh_launch(const std::vector<SshRank>& ranks,
                                       const SshLaunchOptions& opts);

// Build the ssh argv for one rank (exposed for testing / clarity). The remote
// command is passed as a SINGLE argv element (no shell on the launcher side).
std::vector<std::string> build_ssh_argv(const SshRank& r,
                                        const SshLaunchOptions& opts);

// Build the srun argv for one rank: one task on exactly one allocated node,
// sharing the batch step's allocation (--overlap, Slurm >= 20.11), running the
// remote command through /bin/sh -c as ONE argv element (same contract as ssh).
std::vector<std::string> build_srun_argv(const SshRank& r,
                                         const SshLaunchOptions& opts);

// Dispatch to build_ssh_argv or build_srun_argv based on opts.launcher. argv[0]
// is the launcher binary name ("ssh" or "srun"), also used by ssh_launch to
// execvp(2) the right program.
std::vector<std::string> build_launch_argv(const SshRank& r,
                                           const SshLaunchOptions& opts);

// Single-quote a value for safe inclusion in the remote shell command string
// (used for glob-bearing values like --filter 'syn2_*'). Wraps in single quotes
// and escapes embedded single quotes the POSIX way ('\'').
std::string shell_single_quote(const std::string& v);

// The launcher-side run options that shape the remote WORKER command. This is a
// pared-down view of main.cc's Options (only the fields build_remote_command
// needs), lifted here so the builder is linkable/testable outside main.cc.
struct RemoteCommandParams {
  std::string policy;            // --policy file (relative => made ABSOLUTE)
  std::string filter;            // --filter glob (single-quoted in the command)
  std::string filter_class;      // --filter-class
  std::string filter_category;   // --filter-category glob (single-quoted)
  bool        hide_na = false;   // --hide-na
  std::string kind;              // --kind func|perf|all
  std::string isolate;           // --isolate value (only if isolate_explicit)
  bool        isolate_explicit = false;
  std::string gid;               // --gid index override (ssh workers don't inherit the launcher env)
  std::string device;            // --device name override (ssh workers don't inherit the launcher env)
  long        deadline_sec = 0;  // --deadline-sec N (worker self-destruct deadline); 0 => omit the
                                  // flag entirely, so an older --remote-binary that predates
                                  // --deadline-sec is not handed a flag it would reject.
  // Launcher environment variables to forward to the remote worker
  // (--forward-env), already resolved to concrete name/value pairs and sorted
  // by name for deterministic output. Empty => no `env ...` prefix is added
  // (identical command to before env forwarding). Filled for the ssh launcher
  // only — srun passes the caller's whole environment itself (--export=ALL).
  std::vector<std::pair<std::string, std::string>> env;
};

// Build the remote WORKER command for one rank as ONE shell string. Glob-bearing
// values (--filter) are single-quoted so the remote shell does not expand them;
// --policy is resolved to an ABSOLUTE path (the remote worker runs in the ssh
// home dir, not the launcher's cwd) and single-quoted; other values are simple
// tokens (host:port, integers, identifiers). Behavior is identical to the former
// main.cc-local builder.
std::string build_remote_command(const RemoteCommandParams& p, int rank, int size,
                                 const std::string& remote_binary,
                                 const std::string& rendezvous,
                                 const std::string& vendor);

// Split a comma-separated node list, trimming surrounding whitespace and dropping
// empty entries ("a, b," -> ["a","b"]; a single host is fine).
std::vector<std::string> split_nodes(const std::string& csv);

// --- --forward-env --------------------------------------------------
//
// --forward-env LIST is a comma-separated list of environment variable names
// forwarded from the launcher to the ssh worker. A trailing '*' turns an
// entry into a prefix match (e.g. "ACCEPT_*" matches "ACCEPT_IB_DEVICE" but
// not "ACCEPTX"). Only variables actually set in the launcher's own
// environment are ever forwarded.

// Split a comma-separated --forward-env spec, trimming whitespace and
// dropping empty entries (same trimming rules as split_nodes; "" -> no
// entries, i.e. nothing is forwarded).
std::vector<std::string> parse_forward_env_spec(const std::string& csv);

// True if `pattern` is syntactically valid: an identifier
// ([A-Za-z_][A-Za-z0-9_]*), optionally followed by one trailing '*' for a
// prefix match. A bare "*" (empty prefix) is rejected -- forwarding
// "everything" is not an allowed pattern, keeping the set explicit.
bool is_valid_forward_env_pattern(const std::string& pattern);

// True if `name` matches `pattern`: exact match for a bare identifier, or a
// prefix match when `pattern` ends in '*' (the '*' is not itself a regex
// wildcard -- it only ever means "ends the literal prefix here").
bool forward_env_pattern_matches(const std::string& pattern,
                                 const std::string& name);

// Resolve `patterns` against `environment` (e.g. the launcher's own
// environ, as name/value pairs): keep every variable whose name matches at
// least one pattern. A variable matching more than one pattern (e.g. an
// explicit "FOO" entry and a "FOO_*"-style prefix, or two overlapping
// prefixes) is included exactly once. The result is sorted by name for a
// deterministic remote command.
std::vector<std::pair<std::string, std::string>> select_forward_env(
    const std::vector<std::pair<std::string, std::string>>& environment,
    const std::vector<std::string>& patterns);

}  // namespace acceptance
