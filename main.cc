/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// hva — standalone HIP/Verbs acceptance harness.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cerrno>
#include <memory>
#include <sstream>
#include <system_error>

#include <fnmatch.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "common/Bootstrap.hpp"
#include "common/Capability.hpp"
#include "common/Policy.hpp"
#include "common/PosixUtil.hpp"
#include "common/Reporter.hpp"
#include "common/ResultIO.hpp"
#include "common/Runner.hpp"
#include "common/SockCompat.hpp"
#include "common/SocketBootstrap.hpp"
#include "common/SshLauncher.hpp"
#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"
#include "wrapper/hip/api_wrapper.hpp"
#ifdef ACCEPT_HAVE_IBVERBS
#include "wrapper/verbs/ibv_api_wrapper.hpp"
#endif
#include "common/TestResult.hpp"
#include "common/Types.hpp"

#include "hip/utils/HipApi.hpp"

// POSIX environ, used by --forward-env to read the launcher's own
// environment. Not guaranteed to be declared by <unistd.h> on every libc, so
// declare it explicitly.
extern char** environ;

namespace acceptance {

namespace {

// waitpid_eintr / sleep_ms / monotonic_ms live in common/PosixUtil.hpp (shared).
using namespace posix_util;

constexpr const char* kVersion = "hva 0.1.0-phase1";

// Default --forward-env list: forwarded to ssh workers unless
// --no-forward-env is given or --forward-env overrides it. HSA_* and
// ACCEPT_* are prefix matches (trailing '*'); everything else is an exact
// variable name.
constexpr const char* kDefaultForwardEnv =
    "LD_LIBRARY_PATH,ROCM_PATH,PATH,HIP_VISIBLE_DEVICES,ROCR_VISIBLE_DEVICES,"
    "HSA_*,ACCEPT_*";

enum class Mode { None, Help, Version, List, Exec, Run, Worker, LocalRanks, Nodes };

struct Options {
  Mode mode = Mode::None;

  // --exec <id> --result-file <path>
  std::string exec_id;
  std::string result_file;

  // run/worker selection
  std::string filter;              // --filter <glob> on test id
  std::string filter_class;        // --filter-class <class>
  std::string filter_category;     // --filter-category <glob> on category field
  bool        hide_na = false;     // --hide-na: suppress NA rows from console
  std::string kind = "func";       // --kind func|perf|all
  std::string isolate = "all";     // --isolate all|none
  bool isolate_explicit = false;   // true when --isolate was passed
  std::string policy;              // --policy <file> (direct CSV path)
  bool policy_explicit = false;    // true when --policy was passed
  std::string report;              // --report <file>
  std::string perf_csv;            // --perf-csv <file> (metrics as CSV)
  std::string vendor;              // --vendor <name> (overrides auto-detection)
  std::string gid;                 // --gid <index> (override RoCE GID index)
  std::string device;              // --device <name> (prefer this RDMA device)
  std::string results_out;         // --results-out <path>
  bool quiet = false;              // --quiet (summary only; no per-test console lines)
  bool lint = false;               // --lint: report unlisted tests, no run

  // Multi-rank (Step 5).
  bool        worker = false;       // --worker (run one rank, write per-rank file)
  int         rank = 0;             // --rank N
  int         size = 1;             // --size M
  std::string rendezvous;           // --rendezvous HOST:PORT
  int         local_ranks = 0;      // --local-ranks N (launcher; only 2 supported)

  // --nodes launcher: ssh or srun (Phase 1b / Step 6).
  std::string nodes;                // --nodes h1,h2 (comma list; size 1 or 2)
  int         port = 18620;         // --port (rendezvous port on nodes[0])
  std::string ssh_user;             // --ssh-user U (ssh -l U)
  std::string remote_binary;        // --remote-binary PATH (default /proc/self/exe)
  long        launch_timeout = 120; // --launch-timeout SEC (overall deadline)
  RemoteLauncher remote_launcher = RemoteLauncher::Ssh;  // --remote-launcher ssh|srun

  // Env forwarding to ssh workers. ON by default: a trailing '*' in an
  // entry is a prefix match. NOTE: this is a deliberate behaviour change --
  // ssh workers used to start with only the remote login environment; now
  // they additionally inherit this list from the launcher's own environment
  // (values actually set only). --no-forward-env restores the old behaviour.
  std::string forward_env = kDefaultForwardEnv;  // --forward-env LIST
  bool        forward_env_explicit = false;      // true when --forward-env was passed
  bool        no_forward_env = false;             // --no-forward-env
};

void print_usage(std::FILE* out) {
  std::fprintf(out,
    "hva " "0.1.0-phase1" " — HIP/Verbs acceptance harness\n"
    "\n"
    "Usage:\n"
    "  hva --list\n"
    "  hva --exec <id> --result-file <path>          # one test, in-process\n"
    "  hva [--filter <glob>] [--kind func|perf|all]  # worker + report (single-node)\n"
    "             [--filter-class <class>] [--filter-category <glob>] [--hide-na]\n"
    "             [--isolate all|none] [--policy <file>] [--vendor <name>]\n"
    "             [--report <file>] [--results-out <path>]\n"
    "  hva --lint [--policy <file>] [--vendor <name>]  # check coverage\n"
    "  hva --local-ranks 2 [--filter ...] [--policy ...] [--vendor ...] [--report ...]\n"
    "                                                        # spawn 2 local ranks, reconcile\n"
    "  hva --nodes h1,h2 [--remote-launcher ssh|srun] [--port P] [--ssh-user U]\n"
    "             [--remote-binary PATH] [--filter ...] [--policy FILE]\n"
    "             [--vendor V] [--report R]\n"
    "                                                        # launch 1 worker/node (ssh or\n"
    "                                                        # srun), reconcile\n"
    "  hva --worker --rank N --size M --rendezvous HOST:PORT --results-out FILE\n"
    "                                                        # one rank (spawned by launcher)\n"
    "                                                        # --results-out - writes results to\n"
    "                                                        # stdout, logs to stderr\n"
    "\n"
    "Meta:\n"
    "  -h, --help        Show this help and exit\n"
    "      --version     Print version and exit\n"
    "      --list        List registered tests and exit\n"
    "\n"
    "Options:\n"
    "  --exec <id>           Run a single test in-process (EXEC mode)\n"
    "  --result-file <path>  Where EXEC writes its TestResult (key=value).\n"
    "                        Omit to write the result to stdout (debug only;\n"
    "                        the Runner always passes a real path).\n"
    "  --filter <glob>         Select tests by id glob\n"
    "  --filter-class <class>  Select by policy class (must_pass|known_failure|skip|na)\n"
    "  --filter-category <g>   Select by category field glob (e.g. 'hip/*')\n"
    "  --hide-na               Suppress NA rows from console output\n"
    "  --kind func|perf|all    Test kind to run (default: func)\n"
    "  --isolate all|none      Process-isolation scope (default: all)\n"
    "  --policy <file>         Direct path to a policy CSV file. Auto-selected from\n"
    "                          <binary_dir>/policy/<gpu>_<nic>.csv (else <nic>.csv)\n"
    "                          when omitted.\n"
    "  --vendor <name>         Override vendor auto-detection (affects auto policy\n"
    "                          file selection; lower priority than --policy)\n"
    "  --gid <index>           Force the RoCE GID index (override auto-selection)\n"
    "  --device <name>         Prefer this RDMA device (e.g. bnxt_re0 / rdma0)\n"
    "  --report <file>         Where to write report.json (default: CWD)\n"
    "  --perf-csv <file>       Also write the measured metrics as CSV (needs\n"
    "                          --kind perf|all to have anything to write)\n"
    "  --quiet                 Print only the summary (suppress per-test lines)\n"
    "  --lint                  Report registered tests absent from the policy;\n"
    "                          exit 1 if any found. Does not run tests.\n"
    "  --results-out <path>    Write raw per-worker results (debug)\n"
    "  --local-ranks <n>     Spawn n local ranks (only 2 supported) and reconcile\n"
    "  --nodes h1[,h2]       launch one worker per node (1 or 2 nodes; ssh or\n"
    "                        srun, see --remote-launcher) and reconcile into one\n"
    "                        report.json. Mutually exclusive with --local-ranks.\n"
    "  --port <p>            Rendezvous port on nodes[0] (default 18620)\n"
    "  --remote-launcher ssh|srun  How to start each node's worker (default: ssh).\n"
    "                        srun requires running inside a Slurm allocation\n"
    "                        (SLURM_JOB_ID set) and is incompatible with --ssh-user;\n"
    "                        it needs Slurm >= 20.11 for --overlap.\n"
    "  --ssh-user <u>        ssh login user (ssh -l u); only valid with\n"
    "                        --remote-launcher ssh\n"
    "  --remote-binary PATH  Remote hva binary path (default: this\n"
    "                        binary's /proc/self/exe). SAME-PATH ASSUMPTION: the\n"
    "                        binary AND the --policy dir must be reachable at the\n"
    "                        SAME absolute path on every node (NFS/shared FS).\n"
    "  --launch-timeout <s>  Overall launch deadline in seconds (default 120)\n"
    "  --forward-env <list>  Comma-separated env var names forwarded from the\n"
    "                        launcher to each ssh worker; a trailing '*' is a\n"
    "                        prefix match (e.g. 'ACCEPT_*'). Only variables set\n"
    "                        in the launcher's own environment are forwarded.\n"
    "                        Default: LD_LIBRARY_PATH,ROCM_PATH,PATH,\n"
    "                        HIP_VISIBLE_DEVICES,ROCR_VISIBLE_DEVICES,HSA_*,\n"
    "                        ACCEPT_* (ON by default; ssh launcher only).\n"
    "  --no-forward-env      Disable env forwarding (old ssh-login-env behaviour;\n"
    "                        ssh launcher only)\n"
    "  --worker              Run one rank (spawned by the launcher)\n"
    "  --rank <n>            This worker's rank\n"
    "  --size <n>            Total rank count (1 or 2)\n"
    "  --rendezvous H:P      Rank-0 listener host:port for the socket bootstrap\n"
    "  --results-out PATH    Worker results file; '-' = stdout (logs -> stderr)\n");
}

// Fetch the value for a flag that requires an argument. Advances `i`.
// Returns false (and prints an error) when the value is missing.
bool take_value(int argc, char** argv, int& i, const char* flag, std::string& out) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "error: missing value for %s\n", flag);
    return false;
  }
  out = argv[++i];
  return true;
}

// Strict base-10 integer parse (rejects trailing junk; unlike atoi).
bool take_int(int argc, char** argv, int& i, const char* flag, long& out) {
  std::string v;
  if (!take_value(argc, argv, i, flag, v)) return false;
  try {
    std::size_t pos = 0;
    const long n = std::stol(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("trailing characters");
    out = n;
  } catch (const std::exception&) {
    std::fprintf(stderr, "error: %s expects an integer, got '%s'\n", flag,
                 v.c_str());
    return false;
  }
  return true;
}

// Parse argv into Options. Returns 0 on success, or a process exit code
// (2 for usage errors) on failure.
int parse_args(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];

    if (a == "-h" || a == "--help") {
      opt.mode = Mode::Help;
      return 0;  // help short-circuits everything else
    }
    if (a == "--version") {
      opt.mode = Mode::Version;
      return 0;
    }
    if (a == "--list") {
      opt.mode = Mode::List;
    } else if (a == "--exec") {
      if (!take_value(argc, argv, i, "--exec", opt.exec_id)) return 2;
      opt.mode = Mode::Exec;
    } else if (a == "--result-file") {
      if (!take_value(argc, argv, i, "--result-file", opt.result_file)) return 2;
    } else if (a == "--filter") {
      if (!take_value(argc, argv, i, "--filter", opt.filter)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--filter-class") {
      if (!take_value(argc, argv, i, "--filter-class", opt.filter_class)) return 2;
      if (!class_is_valid(opt.filter_class)) {
        std::fprintf(stderr,
                     "error: --filter-class must be one of "
                     "must_pass|known_failure|skip|na (got '%s')\n",
                     opt.filter_class.c_str());
        return 2;
      }
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--filter-category") {
      if (!take_value(argc, argv, i, "--filter-category", opt.filter_category)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--hide-na") {
      opt.hide_na = true;
    } else if (a == "--lint") {
      opt.lint = true;
      opt.mode = Mode::Run;  // triggers policy load; actual run is skipped
    } else if (a == "--kind") {
      if (!take_value(argc, argv, i, "--kind", opt.kind)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--isolate") {
      if (!take_value(argc, argv, i, "--isolate", opt.isolate)) return 2;
      if (opt.isolate != "all" && opt.isolate != "none") {
        std::fprintf(stderr,
                     "error: --isolate must be 'all' or 'none' (got '%s')\n",
                     opt.isolate.c_str());
        return 2;
      }
      opt.isolate_explicit = true;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else if (a == "--policy") {
      if (!take_value(argc, argv, i, "--policy", opt.policy)) return 2;
      opt.policy_explicit = true;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--report") {
      if (!take_value(argc, argv, i, "--report", opt.report)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--perf-csv") {
      if (!take_value(argc, argv, i, "--perf-csv", opt.perf_csv)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--vendor") {
      if (!take_value(argc, argv, i, "--vendor", opt.vendor)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--gid") {
      long v = 0;
      if (!take_int(argc, argv, i, "--gid", v)) return 2;
      if (v < 0) {
        std::fprintf(stderr, "error: --gid must be >= 0 (got %ld)\n", v);
        return 2;
      }
      opt.gid = std::to_string(v);
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--device") {
      if (!take_value(argc, argv, i, "--device", opt.device)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--results-out") {
      if (!take_value(argc, argv, i, "--results-out", opt.results_out)) return 2;
      if (opt.mode == Mode::None) opt.mode = Mode::Run;
    } else if (a == "--worker") {
      opt.worker = true;
      opt.mode = Mode::Worker;
    } else if (a == "--rank") {
      long v = 0;
      if (!take_int(argc, argv, i, "--rank", v)) return 2;
      opt.rank = static_cast<int>(v);
    } else if (a == "--size") {
      long v = 0;
      if (!take_int(argc, argv, i, "--size", v)) return 2;
      opt.size = static_cast<int>(v);
    } else if (a == "--rendezvous") {
      if (!take_value(argc, argv, i, "--rendezvous", opt.rendezvous)) return 2;
    } else if (a == "--local-ranks") {
      long v = 0;
      if (!take_int(argc, argv, i, "--local-ranks", v)) return 2;
      opt.local_ranks = static_cast<int>(v);
      opt.mode = Mode::LocalRanks;
    } else if (a == "--nodes") {
      if (!take_value(argc, argv, i, "--nodes", opt.nodes)) return 2;
      opt.mode = Mode::Nodes;
    } else if (a == "--port") {
      long v = 0;
      if (!take_int(argc, argv, i, "--port", v)) return 2;
      opt.port = static_cast<int>(v);
    } else if (a == "--ssh-user") {
      if (!take_value(argc, argv, i, "--ssh-user", opt.ssh_user)) return 2;
    } else if (a == "--remote-launcher") {
      std::string v;
      if (!take_value(argc, argv, i, "--remote-launcher", v)) return 2;
      if (v == "ssh") {
        opt.remote_launcher = RemoteLauncher::Ssh;
      } else if (v == "srun") {
        opt.remote_launcher = RemoteLauncher::Srun;
      } else {
        std::fprintf(stderr,
                     "error: --remote-launcher must be 'ssh' or 'srun' (got "
                     "'%s')\n",
                     v.c_str());
        return 2;
      }
    } else if (a == "--remote-binary") {
      if (!take_value(argc, argv, i, "--remote-binary", opt.remote_binary)) return 2;
    } else if (a == "--launch-timeout") {
      if (!take_int(argc, argv, i, "--launch-timeout", opt.launch_timeout)) return 2;
    } else if (a == "--forward-env") {
      if (!take_value(argc, argv, i, "--forward-env", opt.forward_env)) return 2;
      opt.forward_env_explicit = true;
    } else if (a == "--no-forward-env") {
      opt.no_forward_env = true;
    } else {
      std::fprintf(stderr, "error: unknown flag '%s'\n\n", a.c_str());
      print_usage(stderr);
      return 2;
    }
  }

  // --nodes (ssh/srun launcher) and --local-ranks (localhost launcher) are mutually
  // exclusive: they are two different launch backends for the same merge path.
  if (!opt.nodes.empty() && opt.local_ranks != 0) {
    std::fprintf(stderr,
                 "error: --nodes and --local-ranks are mutually exclusive\n");
    return 2;
  }

  // --forward-env / --no-forward-env. Reject the combination outright
  // rather than picking a silent last-flag-wins precedence: the two flags
  // express opposite intents (forward this custom list / forward nothing),
  // and a CLI that accepted both would make rerunning a logged command with
  // one extra flag appended silently change behaviour.
  if (opt.forward_env_explicit && opt.no_forward_env) {
    std::fprintf(stderr,
                 "error: --forward-env and --no-forward-env are mutually "
                 "exclusive\n");
    return 2;
  }
  if (!opt.no_forward_env) {
    for (const std::string& pat : parse_forward_env_spec(opt.forward_env)) {
      if (!is_valid_forward_env_pattern(pat)) {
        std::fprintf(stderr,
                     "error: --forward-env: invalid variable name '%s' "
                     "(expected [A-Za-z_][A-Za-z0-9_]*, optionally ending in "
                     "'*')\n",
                     pat.c_str());
        return 2;
      }
    }
  }
  // --remote-launcher srun validation: srun has no login-user concept (it runs
  // inside an existing allocation as the submitting user) and only makes sense
  // inside a Slurm allocation. It always passes the launcher's whole
  // environment (--export=ALL), so an explicit forward list or "forward
  // nothing" cannot be honoured; reject them rather than silently ignore them.
  if (opt.remote_launcher == RemoteLauncher::Srun) {
    if (!opt.ssh_user.empty()) {
      std::fprintf(stderr,
                   "error: --remote-launcher srun does not accept --ssh-user "
                   "(srun has no login-user option)\n");
      return 2;
    }
    if (opt.forward_env_explicit || opt.no_forward_env) {
      std::fprintf(stderr,
                   "error: --remote-launcher srun does not accept "
                   "--forward-env/--no-forward-env (srun always passes the "
                   "launcher's whole environment)\n");
      return 2;
    }
    if (std::getenv("SLURM_JOB_ID") == nullptr) {
      std::fprintf(stderr,
                   "error: --remote-launcher srun must run inside a Slurm "
                   "allocation\n");
      return 2;
    }
  }
  return 0;
}

// --list: one line per registered test with its metadata.
int do_list() {
  for (const auto& m : TestRegistry::all()) {
    std::printf("%s  [%s, %s, %s, timeout=%ds]\n",
                m.id.c_str(),
                m.category.c_str(),
                to_string(m.topology),
                to_string(m.kind),
                m.timeout_sec);
  }
  return 0;
}

int do_exec(const Options& opt) {
  const TestMeta* meta = TestRegistry::find(opt.exec_id);
  if (meta == nullptr) {
    std::fprintf(stderr, "error: no such test '%s'\n", opt.exec_id.c_str());
    return 2;
  }

  LoopbackBootstrap boot;
  TestContext ctx;
  ctx.rank = boot.rank();
  ctx.size = boot.size();
  ctx.boot = &boot;

  const auto t0 = std::chrono::steady_clock::now();
  TestResult result;
  if (acceptance::hipw::hip_category_unavailable(meta->category)) {
    result = skip("HIP runtime unavailable (libamdhip64.so not loaded)");
  } else {
    result = acceptance::run_with_probe([&] {
      try {
        return meta->fn(ctx);
      } catch (const acceptance::hipw::MissingSymbol& e) {
        return skip(std::string("requires ") + e.name);
      }
    });
  }
  const auto t1 = std::chrono::steady_clock::now();

  result.id = meta->id;
  result.duration_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  // TEST HOOK: ACCEPT_SYN_CORRUPT drives syn_corrupt_result corrupt-file path.
  if (!opt.result_file.empty() && std::getenv("ACCEPT_SYN_CORRUPT") != nullptr) {
    std::ofstream bad(opt.result_file, std::ios::binary | std::ios::trunc);
    bad << "@@@ this is not a valid key=value record \x01\x02\n";
    bad.flush();
    return 0;  // exit 0 -> Runner trusts exit code, then finds corrupt file
  }

  if (opt.result_file.empty()) {
    // Debug: --exec without --result-file writes to stdout (--result-file != "-").
    result.to_kv(std::cout);
    return 0;
  }

  std::ofstream out(opt.result_file, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "error: cannot open result file '%s'\n",
                 opt.result_file.c_str());
    return 3;
  }
  result.to_kv(out);
  out.flush();
  if (!out) {
    std::fprintf(stderr, "error: failed writing result file '%s'\n",
                 opt.result_file.c_str());
    return 3;
  }
  return 0;
}

std::vector<TestMeta> select_tests(const Options& opt, bool exclude_two_node) {
  std::vector<TestMeta> sel;
  for (const auto& m : TestRegistry::all()) {
    if (exclude_two_node && m.topology == Topology::TwoNode) continue;
    if (!opt.filter.empty() &&
        ::fnmatch(opt.filter.c_str(), m.id.c_str(), 0) != 0) continue;
    if (!opt.filter_category.empty() &&
        ::fnmatch(opt.filter_category.c_str(), m.category.c_str(), 0) != 0) continue;
    // Perf is folded into the functional, tests no dedicated perf tests,
    // perf|all run them all; only --kind func filters to only functional tests.
    if (opt.kind == "func" && m.kind != Kind::Functional) continue;
    sel.push_back(m);
  }
  return sel;
}

size_t eligible_count(bool exclude_two_node) {
  size_t n = 0;
  for (const auto& m : TestRegistry::all()) {
    if (exclude_two_node && m.topology == Topology::TwoNode) continue;
    ++n;
  }
  return n;
}

constexpr const char* kDefaultReportPath = "./report.json";

// Directory containing the running binary (from /proc/self/exe readlink).
// Falls back to "." on failure (unlikely on Linux GPU clusters).
std::string binary_dir() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  buf[n] = '\0';
  const std::string path(buf);
  const std::size_t slash = path.rfind('/');
  return (slash == std::string::npos) ? "." : path.substr(0, slash);
}

// Map a detected GPU arch (gcnArchName, e.g. "gfx950:sramecc+:xnack-") to the
// marketing platform key used as a policy-file prefix (e.g. "mi350"). Returns
// "" for arches we do not map, in which case the bare NIC vendor is used.
std::string gpu_platform_key(const std::string& gpu_model) {
  const std::string arch = gpu_model.substr(0, gpu_model.find(':'));
  if (arch == "gfx950") return "mi350";  // MI350 / MI355
  if (arch == "gfx942") return "mi300";  // MI300 / MI325
  if (arch == "gfx90a") return "mi250";  // MI250 / MI210
  return "";
}

// Policy selection priority order:
//  1. --policy <file>  — explicit, used directly
//  2. --vendor <name>  — selects <binary_dir>/policy/<name>.csv
//  3. auto-detected platform -> prefer <binary_dir>/policy/<gpu>_<nic>.csv
//     (e.g. mi350_thor.csv), else fall back to <nic>.csv, else discovery.
std::string resolve_vendor(const Options& opt) {
  if (!opt.vendor.empty()) return opt.vendor;
  const std::string& v = environment().vendor;
  if (v == "connectx" || v == "thor" || v == "ionic") {
    const std::string gpu = gpu_platform_key(environment().gpu_model);
    if (!gpu.empty()) {
      const std::string combined = gpu + "_" + v;  // e.g. "mi350_ionic"
      std::ifstream probe(binary_dir() + "/policy/" + combined + ".csv");
      if (probe.good()) return combined;  // platform-specific policy present
    }
    return v;  // fall back to the bare NIC policy (or discovery if absent)
  }
  return "synthetic";
}

// Resolve the policy file path from options + vendor. Always returns a non-empty
// path; the file may or may not exist. Policy::load() handles the missing-file
// case by returning present=false (discovery), which load_policy_or_fail() turns
// into a WARN + all-MUST_PASS run.
std::string resolve_policy_path(const Options& opt, const std::string& vendor) {
  if (!opt.policy.empty()) return opt.policy;
  return binary_dir() + "/policy/" + vendor + ".csv";
}

bool load_policy_or_fail(const Options& opt, const std::string& vendor,
                         Policy& out, int& rc) {
  const std::string path = resolve_policy_path(opt, vendor);
  try {
    out = Policy::load(path, opt.policy_explicit);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    rc = 2;
    return false;
  }
  if (!out.present) {
    // No policy file → WARN and run all as MUST_PASS with gate active.
    std::fprintf(stderr,
                 "WARN: no policy file found for vendor '%s' (tried: %s); "
                 "running all registered tests as MUST_PASS (gate active).\n",
                 vendor.c_str(), path.c_str());
  }
  return true;
}

std::vector<TestMeta> fill_report_selection(const Options& opt,
                                            const std::string& vendor,
                                            bool discovery,
                                            bool exclude_two_node,
                                            ReportInputs& in) {
  in.vendor = vendor;
  in.discovery = discovery;
  in.quiet = opt.quiet;
  in.perf_csv = opt.perf_csv;
  std::vector<TestMeta> selected = select_tests(opt, exclude_two_node);
  for (const auto& m : selected) in.selected_ids.insert(m.id);
  in.filtered = selected.size() != eligible_count(exclude_two_node);
  return selected;
}

int finalize_report(const Options& opt, const Policy& policy,
                    const ReportInputs& in) {
  const std::string path =
      opt.report.empty() ? std::string(kDefaultReportPath) : opt.report;
  return write_report(policy, in, path);
}

// One static line to be printed before the run (NA/SKIP items classified at
// partition time). Stored per-test so the Runner can interleave them with its
// own results in the correct category order.
struct SkippedLine {
  std::string category;
  std::string line;  // pre-formatted console line
};

struct Partition {
  std::vector<TestMeta>                  to_run;
  std::map<std::string, Classification>  classifications;
  // NA/SKIP lines to emit before the run, in category-sorted order.
  std::vector<SkippedLine>               skipped_lines;
};

Partition classify_and_partition(const std::vector<TestMeta>& selected,
                                 const Policy& policy, const Options& opt,
                                 const EnvironmentInfo& env) {
  Partition p;
  p.to_run.reserve(selected.size());
  const Class filter_cls =
      opt.filter_class.empty() ? Class::Na  // sentinel meaning "no filter"
      : class_from(opt.filter_class);
  const bool has_class_filter = !opt.filter_class.empty();

  for (const auto& m : selected) {
    bool found = false;
    // classify with env for `when` predicates; found=false means
    // unlisted — do not run (policy is the authoritative run list).
    const Classification c = policy.classify(m.id, env, &found);
    if (!found) continue;  // unlisted: silently excluded

    // Apply --filter-class after policy classification.
    if (has_class_filter && c.cls != filter_cls) continue;

    p.classifications[m.id] = c;

    // NA and SKIP tests are not run. Collect their console lines here; the
    // caller flushes them interleaved with runner output so the category
    // grouping stays consistent across both sets of lines.
    if (c.cls == Class::Na || c.cls == Class::Skip) {
      if (!opt.quiet && !(c.cls == Class::Na && opt.hide_na)) {
        const FinalStatus fs = (c.cls == Class::Skip) ? FinalStatus::Skip : FinalStatus::Na;
        p.skipped_lines.push_back({m.category, format_test_line(m.id, fs, c.reason)});
      }
      continue;
    }
    p.to_run.push_back(m);
  }
  return p;
}

int do_run(const Options& opt) {
  const std::string vendor = resolve_vendor(opt);
  Policy policy;
  int rc = 0;
  if (!load_policy_or_fail(opt, vendor, policy, rc)) return rc;
  const bool discovery = !policy.present;

  // --lint: cross-reference registry against policy; print unlisted tests.
  if (opt.lint) {
    const std::vector<TestMeta> all = select_tests(opt, /*exclude_two_node=*/false);
    bool any_unlisted = false;
    std::printf("%-50s  %-20s  %s\n", "TEST ID", "CATEGORY", "STATUS");
    std::printf("%-50s  %-20s  %s\n",
                std::string(50, '-').c_str(),
                std::string(20, '-').c_str(), "--------");
    for (const auto& m : all) {
      bool found = false;
      policy.classify(m.id, EnvironmentInfo{}, &found);
      if (!found) {
        std::printf("%-50s  %-20s  NOT IN POLICY\n",
                    m.id.c_str(), m.category.c_str());
        any_unlisted = true;
      }
    }
    if (!any_unlisted) std::printf("All registered tests are covered by the policy.\n");
    return any_unlisted ? 1 : 0;
  }

  ReportInputs in;
  const std::vector<TestMeta> selected =
      fill_report_selection(opt, vendor, discovery, /*exclude_two_node=*/true, in);
  const Partition part = classify_and_partition(selected, policy, opt, environment());

  RunnerOptions ropts;
  ropts.isolate = opt.isolate;
  // Live per-test streaming to stderr (suppressed by --quiet); NA tests were
  // streamed at skip time, so the live stream covers every selected test.
  ropts.stream = !opt.quiet;
  ropts.discovery = discovery;
  ropts.classifications = part.classifications;
  for (const auto& s : part.skipped_lines)
    ropts.skipped_lines.push_back({s.category, s.line});
  // Single-test debug convenience: when exactly one test runs and --isolate was
  // not given explicitly, run it in the parent process so a debugger/strace can
  // attach directly and crashes surface in-line. An explicit --isolate wins.
  if (part.to_run.size() == 1 && !opt.isolate_explicit) ropts.isolate = "none";
  Runner runner(ropts);
  const std::vector<TestResult> results = runner.run(part.to_run);

  // Optional raw worker-results dump (debug), keyed by --results-out. Uses the
  // same per-rank key=value format as the worker (rank=0 here).
  if (!opt.results_out.empty()) {
    const std::string text = result_io::serialize(0, results);
    if (opt.results_out == "-") {
      std::printf("%s", text.c_str());
    } else {
      std::ofstream out(opt.results_out, std::ios::binary | std::ios::trunc);
      if (!out) {
        std::fprintf(stderr, "error: cannot open results-out '%s'\n",
                     opt.results_out.c_str());
        return 3;
      }
      out << text;
      out.flush();
    }
  }

  // The Runner already streamed the per-test lines live (unless --quiet), so the
  // Reporter prints only the summary. (Single rank => live tag == final tag.)
  in.per_test_already_streamed = !opt.quiet;
  for (const auto& r : results) in.add_result(r.id, r);
  // Single-node, in-process: attach the locally detected environment to rank 0.
  in.nodes.push_back({0, "localhost", environment()});
  return finalize_report(opt, policy, in);
}

// Parse "HOST:PORT" into its parts. Returns false on malformed input.
bool parse_rendezvous(const std::string& s, std::string& host, int& port) {
  const auto colon = s.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= s.size()) {
    return false;
  }
  host = s.substr(0, colon);
  const std::string pstr = s.substr(colon + 1);
  try {
    std::size_t pos = 0;
    const long p = std::stol(pstr, &pos);
    if (pos != pstr.size()) return false;  // trailing junk (e.g. "18620x")
    port = static_cast<int>(p);
  } catch (const std::exception&) {
    return false;
  }
  return port > 0 && port <= 65535;
}

int do_worker(const Options& opt) {
  if (opt.results_out.empty()) {
    std::fprintf(stderr, "error: --worker requires --results-out\n");
    return 2;
  }
  if (opt.size < 1 || opt.size > 2) {
    std::fprintf(stderr, "error: --worker supports size 1 or 2 (got %d)\n",
                 opt.size);
    return 2;
  }

  // Construct bootstrap; failure is a hard error (exit 4).
  std::unique_ptr<Bootstrap> boot;
  try {
    if (opt.size == 1) {
      boot = std::make_unique<LoopbackBootstrap>();
    } else {
      std::string host;
      int port = 0;
      if (!parse_rendezvous(opt.rendezvous, host, port)) {
        std::fprintf(stderr, "error: bad --rendezvous '%s' (want HOST:PORT)\n",
                     opt.rendezvous.c_str());
        return 2;
      }
      boot = std::make_unique<SocketBootstrap>(opt.rank, opt.size, host, port);
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: bootstrap failed (rank %d): %s\n", opt.rank,
                 ex.what());
    return 4;  // 4 = bootstrap failure (hard error)
  }

  const std::string vendor = resolve_vendor(opt);

  // Load policy for isolation only; gating happens in the launcher Reporter.
  Policy policy;
  int rc = 0;
  if (!load_policy_or_fail(opt, vendor, policy, rc)) return rc;
  const bool discovery = !policy.present;

  const std::vector<TestMeta> selected =
      select_tests(opt, /*exclude_two_node=*/opt.size == 1);
  const Partition part = classify_and_partition(selected, policy, opt, environment());

  RunnerOptions ropts;
  ropts.isolate = opt.isolate;
  ropts.stream = !opt.quiet;
  ropts.discovery = discovery;
  ropts.classifications = part.classifications;
  for (const auto& s : part.skipped_lines)
    ropts.skipped_lines.push_back({s.category, s.line});
  ropts.boot = boot.get();   // shared bootstrap for in-process (TwoNode) tests
  ropts.rank = opt.rank;
  ropts.size = opt.size;
  Runner runner(ropts);
  const std::vector<TestResult> results = runner.run(part.to_run);

  // --results-out - writes rank block to stdout (the --nodes launcher, ssh or
  // srun, captures it). Each
  // worker emits its DETECTED environment as an @env block so the launcher can
  // build per-node descriptors from the workers' streams.
  const std::string text =
      result_io::serialize(opt.rank, results, environment());
  if (opt.results_out == "-") {
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fflush(stdout);
    return 0;
  }
  std::ofstream out(opt.results_out, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "error: cannot open results-out '%s'\n",
                 opt.results_out.c_str());
    return 3;
  }
  out << text;
  out.flush();
  if (!out) {
    std::fprintf(stderr, "error: failed writing results-out '%s'\n",
                 opt.results_out.c_str());
    return 3;
  }
  return 0;
}

int pick_free_port() {
  const int fd = sockcompat::tcp_socket();
  if (fd < 0) return 0;
  sockcompat::set_reuseaddr(fd);
  if (sockcompat::bind_host_port(fd, "127.0.0.1", 0) < 0) {
    sockcompat::close_fd(fd);
    return 0;
  }
  const int port = sockcompat::local_port(fd);
  sockcompat::close_fd(fd);
  return port;
}

// Parse a worker's output into `in`. Also captures the worker's `@env` block and
// stores it on the matching rank's NodeInfo.environment (matched by rank).
bool load_rank_results_text(const std::string& text, ReportInputs& in,
                            int force_rank) {
  int rk = 0;
  std::vector<TestResult> results;
  EnvironmentInfo env;
  if (!result_io::parse(text, rk, results, env)) return false;
  if (force_rank >= 0) rk = force_rank;
  for (const auto& r : results) {
    if (r.id.empty()) continue;
    in.rank_results[r.id][rk] = r;
  }
  // Attach the parsed environment to the node with this rank, if present.
  for (auto& n : in.nodes) {
    if (n.rank == rk) {
      n.environment = env;
      break;
    }
  }
  return true;
}

bool load_rank_results(const std::string& path, ReportInputs& in) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream buf;
  buf << f.rdbuf();
  return load_rank_results_text(buf.str(), in, /*force_rank=*/-1);
}

int do_local_ranks(const Options& opt) {
  if (opt.local_ranks != 2) {
    std::fprintf(stderr,
                 "error: --local-ranks supports exactly 2 (got %d)\n",
                 opt.local_ranks);
    return 2;
  }

  const int port = pick_free_port();
  if (port == 0) {
    std::fprintf(stderr, "error: could not find a free rendezvous port\n");
    return 3;
  }
  const std::string rendezvous = "127.0.0.1:" + std::to_string(port);

  // Per-run temp dir for the two per-rank result files.
  char tmpl[] = "/tmp/hva_ranks_XXXXXX";
  const char* dir = ::mkdtemp(tmpl);
  if (dir == nullptr) {
    std::fprintf(stderr, "error: mkdtemp failed: %s\n", std::strerror(errno));
    return 3;
  }
  const std::string tmpdir = dir;
  const std::string r0_path = tmpdir + "/r0.kv";
  const std::string r1_path = tmpdir + "/r1.kv";

  auto cleanup = [&]() {
    ::unlink(r0_path.c_str());
    ::unlink(r1_path.c_str());
    ::rmdir(tmpdir.c_str());
  };

  // Spawn a worker for `rank` with its results path.
  auto spawn = [&](int rank, const std::string& rpath) -> pid_t {
    const pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
      // ---- child: execv self in --worker mode ----
      std::vector<std::string> args = {
        "/proc/self/exe", "--worker",
        "--rank", std::to_string(rank),
        "--size", "2",
        "--rendezvous", rendezvous,
        "--results-out", rpath,
        // Launcher reconciles; silence worker live streaming.
        "--quiet",
      };
      if (!opt.filter.empty())  { args.push_back("--filter");  args.push_back(opt.filter); }
      if (!opt.kind.empty())    { args.push_back("--kind");    args.push_back(opt.kind); }
      if (!opt.vendor.empty())  { args.push_back("--vendor");  args.push_back(opt.vendor); }
      if (!opt.policy.empty())  { args.push_back("--policy");  args.push_back(opt.policy); }
      if (opt.isolate_explicit) { args.push_back("--isolate"); args.push_back(opt.isolate); }
      std::vector<char*> cargv;
      cargv.reserve(args.size() + 1);
      for (auto& a : args) cargv.push_back(a.data());
      cargv.push_back(nullptr);
      char self[] = "/proc/self/exe";
      ::execv(self, cargv.data());
      ::_exit(127);  // exec failed
    }
    return pid;
  };

  const pid_t pid0 = spawn(0, r0_path);
  const pid_t pid1 = spawn(1, r1_path);
  if (pid0 < 0 || pid1 < 0) {
    std::fprintf(stderr, "error: fork failed launching workers\n");
    if (pid0 > 0) { ::kill(pid0, SIGKILL); waitpid_eintr(pid0, nullptr, 0); }
    if (pid1 > 0) { ::kill(pid1, SIGKILL); waitpid_eintr(pid1, nullptr, 0); }
    cleanup();
    return 3;
  }

  // Overall-wait guard: poll for both children, bounded by a generous deadline
  // (rendezvous timeout is ~30s per side; give headroom). On timeout, SIGKILL
  // both and error out — NOTHING hangs forever.
  const long overall_sec = 90;
  const long t0_ms = monotonic_ms();

  int st0 = 0, st1 = 0;
  bool done0 = false, done1 = false;
  bool timed_out = false;
  while (!(done0 && done1)) {
    if (!done0) {
      const pid_t r = waitpid_eintr(pid0, &st0, WNOHANG);
      if (r == pid0) done0 = true;
      else if (r < 0) { done0 = true; st0 = -1; }
    }
    if (!done1) {
      const pid_t r = waitpid_eintr(pid1, &st1, WNOHANG);
      if (r == pid1) done1 = true;
      else if (r < 0) { done1 = true; st1 = -1; }
    }
    if (done0 && done1) break;
    if ((monotonic_ms() - t0_ms) / 1000 >= overall_sec) {
      timed_out = true;
      break;
    }
    sleep_ms(20);
  }

  if (timed_out) {
    std::fprintf(stderr,
                 "error: workers did not complete within %lds — killing "
                 "(possible rendezvous hang)\n", overall_sec);
    if (!done0) { ::kill(pid0, SIGKILL); waitpid_eintr(pid0, &st0, 0); }
    if (!done1) { ::kill(pid1, SIGKILL); waitpid_eintr(pid1, &st1, 0); }
    cleanup();
    return 4;  // bootstrap/launch failure
  }

  // Each worker must have exited 0 with a parseable results file. A nonzero exit
  // or signal -> bootstrap/launch failure (hard error, distinct from a test
  // failing).
  auto exit_ok = [](int st) {
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
  };
  if (!exit_ok(st0) || !exit_ok(st1)) {
    std::fprintf(stderr,
                 "error: a worker exited abnormally (rank0 status=%d, "
                 "rank1 status=%d) — bootstrap/run failure\n", st0, st1);
    cleanup();
    return 4;
  }

  // Reconcile: load both per-rank files into one ReportInputs.
  const std::string vendor = resolve_vendor(opt);
  Policy policy;
  int rc = 0;
  if (!load_policy_or_fail(opt, vendor, policy, rc)) { cleanup(); return rc; }
  const bool discovery = !policy.present;

  // 2-rank run keeps TwoNode tests; the denominator is the full registry.
  ReportInputs in;
  fill_report_selection(opt, vendor, discovery, /*exclude_two_node=*/false, in);
  // Placeholder envs: load_rank_results() parses each worker's @env block (the
  // 3-arg result_io::parse) and overwrites the matching node's environment by
  // rank, so these EnvironmentInfo{} are replaced with the detected descriptors.
  in.nodes.push_back({0, "localhost", EnvironmentInfo{}});
  in.nodes.push_back({1, "localhost", EnvironmentInfo{}});

  if (!load_rank_results(r0_path, in) || !load_rank_results(r1_path, in)) {
    std::fprintf(stderr,
                 "error: a worker results file is missing or unparseable — "
                 "bootstrap/run failure\n");
    cleanup();
    return 4;
  }

  const int gate_rc = finalize_report(opt, policy, in);
  cleanup();
  return gate_rc;
}

// Tail of a captured stderr blob, for diagnostics (last ~2KB).
std::string stderr_tail(const std::string& s) {
  constexpr size_t kMax = 2048;
  if (s.size() <= kMax) return s;
  return "...(truncated)...\n" + s.substr(s.size() - kMax);
}

// Snapshot of this process's own environment as name/value pairs:
// --forward-env resolves its patterns against this, never against the
// remote/ssh-login environment. Malformed entries (no '=', which POSIX
// environ never actually produces) are skipped defensively.
std::vector<std::pair<std::string, std::string>> read_process_environment() {
  std::vector<std::pair<std::string, std::string>> out;
  for (char** e = environ; e && *e; ++e) {
    const std::string kv = *e;
    const auto eq = kv.find('=');
    if (eq == std::string::npos) continue;
    out.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
  }
  return out;
}

// Human-readable name of the selected --remote-launcher, for diagnostics.
const char* launcher_name(RemoteLauncher l) {
  return l == RemoteLauncher::Srun ? "srun" : "ssh";
}

// NODES launcher: ssh or srun one worker per node; same absolute binary+policy
// path required.
int do_nodes(const Options& opt) {
  const std::vector<std::string> nodes = split_nodes(opt.nodes);
  if (nodes.empty()) {
    std::fprintf(stderr, "error: --nodes is empty\n");
    return 2;
  }
  if (nodes.size() > 2) {
    std::fprintf(stderr,
                 "error: --nodes supports 1 or 2 nodes (got %zu); size>2 is not "
                 "supported (SocketBootstrap is scoped to size<=2)\n",
                 nodes.size());
    return 2;
  }
  if (opt.port <= 0 || opt.port > 65535) {
    std::fprintf(stderr, "error: bad --port %d\n", opt.port);
    return 2;
  }

  const int size = static_cast<int>(nodes.size());
  const std::string vendor = resolve_vendor(opt);

  // Remote binary: default to our own absolute path (same-path/NFS assumption).
  std::string remote_binary = opt.remote_binary;
  if (remote_binary.empty()) {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
      std::fprintf(stderr,
                   "error: cannot resolve /proc/self/exe for the remote binary; "
                   "pass --remote-binary PATH\n");
      return 2;
    }
    buf[n] = '\0';
    remote_binary = buf;
  }

  // Rendezvous = nodes[0]:port. rank0 (nodes[0]) is the listener (binds
  // 0.0.0.0); rank1 (nodes[1]) dials nodes[0]:port.
  const std::string rendezvous =
      nodes[0] + ":" + std::to_string(opt.port);

  // Build the per-rank launch specs (ssh or srun argv is built from these).
  RemoteCommandParams rcp;
  rcp.policy          = opt.policy;
  rcp.filter          = opt.filter;
  rcp.filter_class    = opt.filter_class;
  rcp.filter_category = opt.filter_category;
  rcp.hide_na         = opt.hide_na;
  rcp.kind            = opt.kind;
  rcp.isolate         = opt.isolate;
  rcp.isolate_explicit = opt.isolate_explicit;
  rcp.gid             = opt.gid;
  rcp.device          = opt.device;
  // --forward-env: ssh launcher only. srun passes the launcher's whole
  // environment itself (--export=ALL), so an `env` prefix would be redundant;
  // parse_args already rejected the explicit flags with srun.
  if (opt.remote_launcher == RemoteLauncher::Ssh && !opt.no_forward_env) {
    const std::vector<std::string> patterns =
        parse_forward_env_spec(opt.forward_env);
    if (!patterns.empty()) {
      rcp.env = select_forward_env(read_process_environment(), patterns);
    }
  }
  std::vector<SshRank> specs;
  specs.reserve(nodes.size());
  for (int i = 0; i < size; ++i) {
    SshRank s;
    s.rank = i;
    s.node = nodes[i];
    s.remote_command =
        build_remote_command(rcp, i, size, remote_binary, rendezvous, vendor);
    specs.push_back(std::move(s));
  }

  SshLaunchOptions lopts;
  lopts.launcher = opt.remote_launcher;
  lopts.ssh_user = opt.ssh_user;
  lopts.overall_timeout_sec = opt.launch_timeout;
  // Connect timeout: keep it under the overall deadline so a small
  // --launch-timeout isn't inverted by a larger ConnectTimeout. Ssh-only
  // (srun ignores it).
  lopts.connect_timeout = std::min(10, static_cast<int>(opt.launch_timeout));

  const char* const lname = launcher_name(opt.remote_launcher);
  std::fprintf(stderr,
               "launching %d rank(s) over %s; rendezvous=%s binary=%s\n",
               size, lname, rendezvous.c_str(), remote_binary.c_str());

  const std::vector<SshChildResult> children = ssh_launch(specs, lopts);

  // Validate every child: launched, exited 0, no timeout. Any failure -> exit 4
  // with the offending node + a stderr tail for diagnosis.
  for (const auto& c : children) {
    if (!c.launched) {
      std::fprintf(stderr,
                   "error: failed to launch %s for rank %d on %s: %s\n",
                   lname, c.rank, c.node.c_str(), c.stderr_data.c_str());
      return 4;
    }
    if (c.timed_out) {
      std::fprintf(stderr,
                   "error: rank %d on %s did not complete within %lds — killed "
                   "(possible unreachable node / rendezvous hang)\n",
                   c.rank, c.node.c_str(), opt.launch_timeout);
      std::fprintf(stderr, "--- rank %d (%s) stderr tail ---\n%s\n",
                   c.rank, c.node.c_str(), stderr_tail(c.stderr_data).c_str());
      return 4;
    }
    if (!c.exited || c.exit_code != 0) {
      std::fprintf(stderr,
                   "error: rank %d on %s failed (%s/worker exit_code=%d%s) — "
                   "launch/bootstrap failure\n",
                   c.rank, c.node.c_str(), lname, c.exit_code,
                   c.term_signal ? ", killed by signal" : "");
      std::fprintf(stderr, "--- rank %d (%s) stderr tail ---\n%s\n",
                   c.rank, c.node.c_str(), stderr_tail(c.stderr_data).c_str());
      return 4;
    }
  }

  // Load policy for reconciliation/gating (the launcher gates, not the workers).
  Policy policy;
  int rc = 0;
  if (!load_policy_or_fail(opt, vendor, policy, rc)) return rc;
  const bool discovery = !policy.present;

  // A multi-node run keeps TwoNode tests; size-1 excludes them (cannot satisfy a
  // 2-rank test). Mirror the worker's selection scope so the denominator and the
  // synthesized NA set match what actually ran.
  ReportInputs in;
  fill_report_selection(opt, vendor, discovery, /*exclude_two_node=*/size == 1, in);
  for (const auto& c : children)
    in.nodes.push_back({c.rank, c.node, EnvironmentInfo{}});

  // Parse each child's captured stdout as that rank's result block. The launcher
  // knows the rank, so force it (defends against a mismatched rank= header).
  for (const auto& c : children) {
    if (!load_rank_results_text(c.stdout_data, in, /*force_rank=*/c.rank)) {
      std::fprintf(stderr,
                   "error: rank %d on %s produced no parseable result block on "
                   "stdout — launch/run failure\n",
                   c.rank, c.node.c_str());
      std::fprintf(stderr, "--- rank %d (%s) stderr tail ---\n%s\n",
                   c.rank, c.node.c_str(), stderr_tail(c.stderr_data).c_str());
      return 4;
    }
  }

  return finalize_report(opt, policy, in);
}

int run(const Options& opt) {
  switch (opt.mode) {
    case Mode::Help:
      print_usage(stdout);
      return 0;

    case Mode::Version:
      std::printf("%s\n", kVersion);
      return 0;

    case Mode::List:
      return do_list();

    case Mode::Exec:
      return do_exec(opt);

    case Mode::Worker:
      return do_worker(opt);

    case Mode::LocalRanks:
      return do_local_ranks(opt);

    case Mode::Nodes:
      return do_nodes(opt);

    case Mode::Run:
      // Single-node LAUNCH: worker + policy + Reporter -> report.json + gate.
      return do_run(opt);

    case Mode::None:
      // No explicit mode requested => default single-node run over the full
      // registry (the documented default invocation).
      return do_run(opt);
  }
  return 0;
}

}  // namespace
}  // namespace acceptance

int main(int argc, char** argv) {
  // Resolve every HIP/HSA entry point once, eagerly, before any test runs --
  // mirrors oneCCL's global_data::init() -> api_wrappers_init(). Runs in every
  // process the suite spawns (launcher, --worker, and each --exec child).
  (void)acceptance::hipw::init();
#ifdef ACCEPT_HAVE_IBVERBS
  // Same eager model for libibverbs: open it once so verbs tests resolve their
  // ops table (or the whole verbs category skips cleanly if it is absent).
  (void)acceptance::ibvw::ibv_api_init();
#endif

  acceptance::Options opt;
  if (int rc = acceptance::parse_args(argc, argv, opt); rc != 0) {
    return rc;
  }
  // Perf measurement rides --kind (on for perf/all); set once per process.
  acceptance::set_perf_mode(opt.kind == "perf" || opt.kind == "all");
  // Asking for a perf artifact without turning measurement on yields an empty
  // one — say so rather than let the caller find a header-only CSV later.
  if (!opt.perf_csv.empty() && !acceptance::perf_probe_mode())
    std::fprintf(stderr,
                 "warning: --perf-csv without --kind perf|all: no metrics are "
                 "measured, the file will hold only its header\n");
  // Both artifacts are opened with truncate and the CSV is written second, so
  // one path for both silently eats the report. Refuse here rather than at the
  // end of the run: the whole point is not to spend an hour discovering it.
  // Compare resolved, canonicalized paths rather than the raw strings: --report
  // defaults to kDefaultReportPath when omitted (finalize_report resolves it
  // the same way), and a relative path / symlink can name the same file as
  // --perf-csv without being textually equal.
  if (!opt.perf_csv.empty()) {
    const std::string report_path = opt.report.empty()
        ? std::string(acceptance::kDefaultReportPath)
        : opt.report;
    std::error_code csv_ec, report_ec;
    const auto csv_canon = std::filesystem::weakly_canonical(opt.perf_csv, csv_ec);
    const auto report_canon = std::filesystem::weakly_canonical(report_path, report_ec);
    const bool same = (!csv_ec && !report_ec) ? (csv_canon == report_canon)
                                               : (opt.perf_csv == report_path);
    if (same) {
      std::fprintf(stderr,
                   "error: --perf-csv and --report must differ ('%s'): the CSV is "
                   "written last and would overwrite the report\n",
                   report_path.c_str());
      return 2;
    }
  }
  // Publish --gid/--device as env vars BEFORE any detect()/device open. setenv
  // propagates to forked --exec/--worker children automatically (they inherit
  // the environment); --nodes workers (ssh or srun) get them forwarded as flags (see the
  // worker arg builder). Capability reads ACCEPT_IB_GID_INDEX/ACCEPT_IB_DEVICE.
  if (!opt.gid.empty()) ::setenv("ACCEPT_IB_GID_INDEX", opt.gid.c_str(), 1);
  if (!opt.device.empty()) ::setenv("ACCEPT_IB_DEVICE", opt.device.c_str(), 1);
  return acceptance::run(opt);
}
