/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Launch — rank placement and the worker argv for the mpirun-style launcher
// (-n / -ppn / -hosts).
//
// The launcher places ranks on hosts in block order (ranks 0..P-1 on hosts[0],
// and so on), then starts one worker per rank: ranks on this host are
// fork/exec'd directly, ranks on other hosts go through ssh or srun (see
// SshLauncher). Both kinds get the SAME worker argv from build_worker_args(),
// the one place that lists the flags forwarded to a worker.

#include <string>
#include <vector>

namespace acceptance {

// Where one rank runs.
struct RankPlacement {
  int         rank = 0;
  int         local_rank = 0;   // index among the ranks on the same host
  std::string host;             // as given in -hosts ("localhost" by default)
  bool        is_local = false; // host is this machine: fork/exec, no ssh/srun
};

// True when `host` names this machine: "localhost", "127.0.0.1", or the full
// or short form of `self_hostname` (gethostname()). A short `host` also
// matches a fully qualified `self_hostname` and vice versa.
bool host_is_local(const std::string& host, const std::string& self_hostname);

// Ranks per host actually used: `ppn` when given (> 0), else
// ceil(np / #hosts). An empty host list counts as one host (localhost).
int effective_ppn(int np, int ppn, std::size_t num_hosts);

// Check an -n / -ppn / -hosts combination. Returns "" when valid, else the
// error message (without the "error: " prefix). `np_explicit` is whether -n
// was given; `ppn` is 0 when -ppn was not given.
std::string validate_launch(int np, bool np_explicit, int ppn,
                            const std::vector<std::string>& hosts);

// Block placement of `np` ranks over `hosts` (empty => {"localhost"}). Assumes
// validate_launch() accepted the arguments.
std::vector<RankPlacement> plan_ranks(int np, int ppn,
                                      const std::vector<std::string>& hosts,
                                      const std::string& self_hostname);

// The launcher-side options that shape each worker's command line: the
// subset of main.cc's Options a worker needs, so the argv builder is
// testable outside main.cc.
struct WorkerParams {
  std::string policy;            // --policy file (relative => made ABSOLUTE)
  std::string filter;            // --filter glob
  std::string filter_class;      // --filter-class
  std::string filter_category;   // --filter-category glob
  bool        hide_na = false;   // --hide-na
  std::string kind;              // --kind func|perf|all
  std::string isolate;           // --isolate value (only if isolate_explicit)
  bool        isolate_explicit = false;
  std::string gid;               // --gid index override
  std::string hca;               // --hca list (1 or one per local rank)
  std::string device;            // --device GPU list (1 or one per local rank)
  long        deadline_sec = 0;  // --deadline-sec N; 0 => omit the flag
};

// The worker argv for one rank, WITHOUT the binary (argv[0]). The rank gets
// --local-rank and its own --hca / --device entry. Results always
// go to stdout (--results-out -), where the launcher captures them for every
// rank, local or remote. --policy is made absolute: a remote worker starts in
// the ssh home directory, not the launcher's cwd.
std::vector<std::string> build_worker_args(const WorkerParams& p,
                                           const RankPlacement& placement,
                                           int size,
                                           const std::string& rendezvous,
                                           const std::string& vendor);

}  // namespace acceptance
