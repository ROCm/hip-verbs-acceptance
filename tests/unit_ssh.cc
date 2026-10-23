/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_ssh — unit tests for the ssh/srun/CLI builders + quoting that the
// launcher relies on for remote ranks (common/SshLauncher.*). Covers
// shell_single_quote, shell_quote_if_needed, build_ssh_argv,
// build_remote_command (a glob such as --filter 'syn2_*' stays quoted so the
// remote shell can't expand it), build_launch_argv's ssh/srun dispatch, and
// split_nodes. The worker flags themselves are tested in unit_launch.cc.

#include <algorithm>
#include <string>
#include <vector>

#include "common/SshLauncher.hpp"
#include "tests/unit_harness.hpp"

using namespace acceptance;

namespace accept_test {

namespace {

// True if `needle` appears as a contiguous substring of `hay`.
bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// True if `argv` contains an element exactly equal to `s`.
bool has_arg(const std::vector<std::string>& argv, const std::string& s) {
  return std::find(argv.begin(), argv.end(), s) != argv.end();
}

// Index of the first element equal to `s`, or -1.
int index_of(const std::vector<std::string>& argv, const std::string& s) {
  for (size_t i = 0; i < argv.size(); ++i)
    if (argv[i] == s) return static_cast<int>(i);
  return -1;
}

void test_shell_single_quote() {
  // Plain string -> wrapped in single quotes verbatim.
  CHECK_EQ_STR(shell_single_quote("syn2_x"), "'syn2_x'",
               "shell_single_quote plain");
  // Empty string -> empty quotes.
  CHECK_EQ_STR(shell_single_quote(""), "''", "shell_single_quote empty");
  // A glob is preserved literally inside the quotes (the whole point: the remote
  // shell must NOT expand it).
  CHECK_EQ_STR(shell_single_quote("syn2_*"), "'syn2_*'",
               "shell_single_quote glob");
  // An embedded single quote becomes the canonical POSIX escape '\'' :
  //   a'b -> 'a'\''b'
  CHECK_EQ_STR(shell_single_quote("a'b"), "'a'\\''b'",
               "shell_single_quote embedded quote");
}

void test_build_ssh_argv() {
  SshRank r;
  r.rank = 1;
  r.node = "host2";
  r.remote_command = "the remote command --filter 'x*'";

  // No user set: -l absent.
  {
    SshLaunchOptions o;
    o.connect_timeout = 7;
    const std::vector<std::string> a = build_ssh_argv(r, o);
    CHECK(has_arg(a, "-n"), "ssh argv has -n (no stdin)");
    CHECK(has_arg(a, "BatchMode=yes"), "ssh argv BatchMode=yes");
    CHECK(has_arg(a, "StrictHostKeyChecking=no"),
          "ssh argv StrictHostKeyChecking=no");
    CHECK(has_arg(a, "ConnectTimeout=7"), "ssh argv ConnectTimeout from opts");
    CHECK(!has_arg(a, "-l"), "ssh argv has no -l when user empty");
    // node + the remote command are present, and the remote command is ONE argv
    // element (verbatim, not split).
    CHECK(has_arg(a, "host2"), "ssh argv contains node");
    CHECK(has_arg(a, r.remote_command),
          "ssh argv contains remote command as one element");
    // node immediately precedes the single remote-command element.
    const int ni = index_of(a, "host2");
    CHECK(ni >= 0 && ni + 1 == static_cast<int>(a.size()) - 1,
          "node is second-to-last; remote command is last");
    CHECK(a.back() == r.remote_command, "remote command is the last argv slot");
  }

  // User set: -l <user> present, in order.
  {
    SshLaunchOptions o;
    o.ssh_user = "testuser";
    const std::vector<std::string> a = build_ssh_argv(r, o);
    const int li = index_of(a, "-l");
    CHECK(li >= 0, "ssh argv has -l when user set");
    CHECK(li >= 0 && li + 1 < static_cast<int>(a.size()) &&
              a[li + 1] == "testuser",
          "-l is followed by the user");
  }
}

void test_shell_quote_if_needed() {
  // Safe tokens pass through unchanged: flags, integers, host:port, paths.
  CHECK_EQ_STR(shell_quote_if_needed("--worker"), "--worker",
               "quote_if_needed: flag unchanged");
  CHECK_EQ_STR(shell_quote_if_needed("node0:18620"), "node0:18620",
               "quote_if_needed: host:port unchanged");
  CHECK_EQ_STR(shell_quote_if_needed("/abs/policy.csv"), "/abs/policy.csv",
               "quote_if_needed: plain path unchanged");
  CHECK_EQ_STR(shell_quote_if_needed("-"), "-",
               "quote_if_needed: '-' unchanged");
  // Anything the shell would interpret is single-quoted.
  CHECK_EQ_STR(shell_quote_if_needed("syn2_*"), "'syn2_*'",
               "quote_if_needed: glob quoted");
  CHECK_EQ_STR(shell_quote_if_needed("hip/[a-z]*"), "'hip/[a-z]*'",
               "quote_if_needed: bracket glob quoted");
  CHECK_EQ_STR(shell_quote_if_needed("a b"), "'a b'",
               "quote_if_needed: space quoted");
  CHECK_EQ_STR(shell_quote_if_needed("$HOME"), "'$HOME'",
               "quote_if_needed: $ quoted");
  CHECK_EQ_STR(shell_quote_if_needed("a'b"), "'a'\\''b'",
               "quote_if_needed: embedded quote escaped");
  CHECK_EQ_STR(shell_quote_if_needed(""), "''",
               "quote_if_needed: empty string stays an argument");
}

void test_build_remote_command() {
  const std::vector<std::string> args = {
      "--worker", "--rank", "0", "--rendezvous", "node0:18620",
      "--results-out", "-", "--filter", "syn2_*", "--vendor", "acme; rm -rf /"};
  const std::string cmd = build_remote_command({}, "/abs/hva", args);

  CHECK(cmd.rfind("'/abs/hva' --worker --rank 0", 0) == 0,
        "remote cmd: quoted binary, then the args in order");
  CHECK(contains(cmd, "--rendezvous node0:18620"),
        "remote cmd: safe values are not quoted");
  CHECK(contains(cmd, "--results-out -"), "remote cmd routes results to stdout");
  // The glob is SINGLE-QUOTED so the remote shell can't expand it.
  CHECK(contains(cmd, "--filter 'syn2_*'"),
        "remote cmd single-quotes the --filter glob");
  CHECK(!contains(cmd, "--filter syn2_*"),
        "remote cmd does NOT pass the glob unquoted");
  // Values with shell metacharacters are quoted, not interpolated raw — no
  // word-splitting, no injection into the remote shell.
  CHECK(contains(cmd, "--vendor 'acme; rm -rf /'"),
        "vendor with shell metacharacters is single-quoted (no injection)");
  CHECK(!contains(cmd, "--vendor acme; rm"),
        "metacharacters never reach the remote shell unquoted");

  // A binary path with a space stays one word.
  const std::string spaced = build_remote_command({}, "/opt/my tools/hva", {});
  CHECK_EQ_STR(spaced, "'/opt/my tools/hva'",
               "binary path containing a space is single-quoted");
}

void test_build_launch_argv() {
  SshRank r;
  r.rank = 1;
  r.node = "host2";
  r.remote_command = "the remote command --filter 'x*'";

  // Ssh (default): build_launch_argv must be IDENTICAL to build_ssh_argv —
  // no regression for the existing default launcher.
  {
    SshLaunchOptions o;
    o.launcher = RemoteLauncher::Ssh;
    o.ssh_user = "testuser";
    o.connect_timeout = 9;
    const std::vector<std::string> dispatched = build_launch_argv(r, o);
    const std::vector<std::string> direct = build_ssh_argv(r, o);
    CHECK(dispatched == direct,
          "build_launch_argv(Ssh) == build_ssh_argv (no regression)");
  }

  // Srun: argv shape matches the plan's srun invocation.
  {
    SshLaunchOptions o;
    o.launcher = RemoteLauncher::Srun;
    const std::vector<std::string> a = build_launch_argv(r, o);
    CHECK(!a.empty() && a[0] == "srun", "srun argv[0] == 'srun'");
    CHECK(has_arg(a, "--nodelist=host2"), "srun argv has --nodelist=<node>");
    CHECK(has_arg(a, "--ntasks=1"), "srun argv has --ntasks=1");
    CHECK(has_arg(a, "--nodes=1"), "srun argv has --nodes=1");
    CHECK(has_arg(a, "--overlap"), "srun argv has --overlap");
    CHECK(has_arg(a, "--export=ALL"), "srun argv has --export=ALL");
    CHECK(has_arg(a, "--kill-on-bad-exit=1"),
          "srun argv has --kill-on-bad-exit=1");
    // Ends with /bin/sh, -c, <remote_command> as one element (same contract
    // as ssh: the remote command is never split/re-quoted on the way out).
    CHECK(a.size() >= 3 &&
              a[a.size() - 3] == "/bin/sh" &&
              a[a.size() - 2] == "-c" &&
              a.back() == r.remote_command,
          "srun argv ends with /bin/sh, -c, <remote_command> as one element");
    // No ssh-only options leak into the srun argv.
    CHECK(!has_arg(a, "-n"), "srun argv has no ssh -n");
    CHECK(!has_arg(a, "-l"), "srun argv has no ssh -l");
    CHECK(index_of(a, "BatchMode=yes") < 0,
          "srun argv has no ssh BatchMode option");
    CHECK(index_of(a, "StrictHostKeyChecking=no") < 0,
          "srun argv has no ssh StrictHostKeyChecking option");
    for (const auto& e : a)
      CHECK(e.find("ConnectTimeout=") == std::string::npos,
            "srun argv has no ssh ConnectTimeout option");
  }
}

void test_split_nodes() {
  {
    const std::vector<std::string> v = split_nodes("a,b");
    CHECK(v.size() == 2, "split a,b -> 2 entries");
    CHECK(v.size() == 2 && v[0] == "a" && v[1] == "b", "split a,b values");
  }
  {
    // Trims surrounding whitespace and drops empty entries.
    const std::vector<std::string> v = split_nodes(" a , b ,,");
    CHECK(v.size() == 2, "split ' a , b ,,' drops empties -> 2");
    CHECK(v.size() == 2 && v[0] == "a" && v[1] == "b",
          "split trims surrounding whitespace");
  }
  {
    const std::vector<std::string> v = split_nodes("host1");
    CHECK(v.size() == 1 && v[0] == "host1", "single host OK");
  }
  {
    const std::vector<std::string> v = split_nodes("");
    CHECK(v.empty(), "empty string -> no entries");
  }
  {
    const std::vector<std::string> v = split_nodes(" , , ");
    CHECK(v.empty(), "all-empty/whitespace -> no entries");
  }
}

void test_build_remote_command_env_forwarding() {
  const std::vector<std::string> args = {"--worker"};

  // Empty env -> no "env" prefix at all.
  {
    const std::string cmd = build_remote_command({}, "/abs/hva", args);
    CHECK(cmd.rfind("'/abs/hva'", 0) == 0,
          "empty env: command starts with the binary, no env prefix");
    CHECK(!contains(cmd, "env '"), "empty env: no 'env ...' prefix at all");
  }

  // Two vars -> "env 'A=1' 'B=x y' " immediately precedes the binary, in the
  // order given (the launcher sorts before calling in).
  {
    const std::string cmd =
        build_remote_command({{"A", "1"}, {"B", "x y"}}, "/abs/hva", args);
    CHECK(cmd.rfind("env 'A=1' 'B=x y' '/abs/hva'", 0) == 0,
          "two vars: command starts with env 'A=1' 'B=x y' '<binary>'");
  }

  // A value containing a single quote is escaped the canonical POSIX way,
  // and a value containing an embedded newline survives literally (single
  // quotes are the one POSIX quoting form that preserves it).
  {
    const std::string cmd = build_remote_command(
        {{"Q", "a'b"}, {"N", "line1\nline2"}}, "/abs/hva", args);
    CHECK(contains(cmd, "'Q=a'\\''b'"),
          "env value with an embedded quote is escaped correctly");
    CHECK(contains(cmd, "'N=line1\nline2'"),
          "env value with an embedded newline is preserved literally");
  }
}

void test_forward_env_pattern_matches() {
  CHECK(forward_env_pattern_matches("ACCEPT_*", "ACCEPT_IB_DEVICE"),
        "ACCEPT_* matches ACCEPT_IB_DEVICE");
  CHECK(!forward_env_pattern_matches("ACCEPT_*", "ACCEPTX"),
        "ACCEPT_* does not match ACCEPTX (no '_' boundary)");
  CHECK(forward_env_pattern_matches("PATH", "PATH"),
        "bare name matches itself exactly");
  CHECK(!forward_env_pattern_matches("PATH", "PATH2"),
        "bare name does not prefix-match");
  CHECK(forward_env_pattern_matches("HSA_*", "HSA_"),
        "prefix pattern matches the prefix itself (empty suffix)");
}

void test_is_valid_forward_env_pattern() {
  CHECK(is_valid_forward_env_pattern("PATH"), "PATH is valid");
  CHECK(is_valid_forward_env_pattern("_FOO"), "leading underscore is valid");
  CHECK(is_valid_forward_env_pattern("ACCEPT_*"),
        "identifier with trailing '*' is valid");
  CHECK(!is_valid_forward_env_pattern("1BAD"),
        "name starting with a digit is rejected");
  CHECK(!is_valid_forward_env_pattern("A-B"),
        "name containing '-' is rejected");
  CHECK(!is_valid_forward_env_pattern("*"),
        "bare '*' (empty prefix) is rejected");
  CHECK(!is_valid_forward_env_pattern(""), "empty pattern is rejected");
  CHECK(!is_valid_forward_env_pattern("A*B"),
        "'*' only valid as the trailing character");
}

void test_select_forward_env() {
  const std::vector<std::pair<std::string, std::string>> env = {
      {"ZZZ", "last"},
      {"PATH", "/bin:/usr/bin"},
      {"HSA_OVERRIDE_GFX_VERSION", "11.0.0"},
      {"UNRELATED", "nope"},
  };

  // Only variables that are actually set AND match a pattern are forwarded;
  // output is sorted by name regardless of input/pattern order.
  {
    const auto sel = select_forward_env(env, {"PATH", "HSA_*"});
    CHECK(sel.size() == 2, "select: PATH + HSA_* -> 2 entries");
    CHECK(sel.size() == 2 && sel[0].first == "HSA_OVERRIDE_GFX_VERSION" &&
              sel[1].first == "PATH",
          "select: sorted by name (HSA_* before PATH)");
  }

  // A name that matches BOTH an explicit entry and a prefix pattern is
  // forwarded exactly once (duplicate pattern coverage, not duplicate var).
  {
    const auto sel =
        select_forward_env(env, {"HSA_OVERRIDE_GFX_VERSION", "HSA_*"});
    CHECK(sel.size() == 1, "select: overlapping explicit+prefix -> 1 entry");
    CHECK(sel.size() == 1 && sel[0].first == "HSA_OVERRIDE_GFX_VERSION" &&
              sel[0].second == "11.0.0",
          "select: de-duplicated entry keeps the right value");
  }

  // A pattern matching nothing set in the environment yields nothing (not
  // an error at this layer -- validation happens on the pattern itself).
  {
    const auto sel = select_forward_env(env, {"NOT_SET_ANYWHERE"});
    CHECK(sel.empty(), "select: pattern matching no set var -> empty");
  }

  // Empty pattern list -> nothing forwarded (mirrors --no-forward-env /
  // --forward-env '').
  {
    const auto sel = select_forward_env(env, {});
    CHECK(sel.empty(), "select: empty pattern list -> empty");
  }
}

void test_parse_forward_env_spec() {
  CHECK(parse_forward_env_spec("").empty(),
        "parse_forward_env_spec(''): no entries (nothing forwarded)");
  {
    const auto v = parse_forward_env_spec("A,,B");
    CHECK(v.size() == 2 && v[0] == "A" && v[1] == "B",
          "parse_forward_env_spec: drops the empty middle entry");
  }
  {
    const auto v = parse_forward_env_spec(" A_* , B ");
    CHECK(v.size() == 2 && v[0] == "A_*" && v[1] == "B",
          "parse_forward_env_spec: trims surrounding whitespace");
  }
}

}  // namespace

void run_ssh_tests() {
  test_shell_single_quote();
  test_shell_quote_if_needed();
  test_build_ssh_argv();
  test_build_remote_command();
  test_build_remote_command_env_forwarding();
  test_forward_env_pattern_matches();
  test_is_valid_forward_env_pattern();
  test_select_forward_env();
  test_parse_forward_env_spec();
  test_build_launch_argv();
  test_split_nodes();
}

}  // namespace accept_test
