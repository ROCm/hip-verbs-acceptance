/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// unit_ssh — unit tests for the ssh/CLI builders + quoting that the --nodes
// launcher relies on (common/SshLauncher.*). Covers shell_single_quote,
// build_ssh_argv, build_remote_command (incl. --policy made ABSOLUTE & quoted
// and --filter single-quoted so a glob can't be expanded), and split_nodes.

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

void test_build_remote_command() {
  RemoteCommandParams p;
  p.policy          = "policy";     // RELATIVE -> must be made absolute
  p.filter          = "syn2_*";     // glob -> must be single-quoted
  p.filter_class    = "must_pass";
  p.filter_category = "hip/*";      // glob -> must be single-quoted
  p.hide_na         = true;
  p.kind            = "func";
  p.isolate         = "all";
  p.isolate_explicit = true;

  const std::string cmd = build_remote_command(
      p, /*rank=*/0, /*size=*/2, "/abs/hva", "node0:18620", "synthetic");

  // All interpolated values are single-quoted (shell-injection / spaces guard).
  CHECK(contains(cmd, "'/abs/hva'"),
        "remote cmd single-quotes the binary path");
  CHECK(contains(cmd, "--worker"), "remote cmd has --worker");
  CHECK(contains(cmd, "--rank 0"), "remote cmd has --rank 0");
  CHECK(contains(cmd, "--size 2"), "remote cmd has --size 2");
  CHECK(contains(cmd, "--rendezvous 'node0:18620'"),
        "remote cmd single-quotes --rendezvous host:port");
  CHECK(contains(cmd, "--results-out -"),
        "remote cmd routes results to stdout");
  CHECK(contains(cmd, "--vendor 'synthetic'"), "remote cmd single-quotes --vendor");
  CHECK(contains(cmd, "--kind 'func'"), "remote cmd single-quotes --kind");
  CHECK(contains(cmd, "--isolate 'all'"),
        "remote cmd single-quotes --isolate when explicit");

  // --filter / --filter-category are SINGLE-QUOTED so the remote shell can't
  // expand the globs.
  CHECK(contains(cmd, "--filter 'syn2_*'"),
        "remote cmd single-quotes the --filter glob");
  CHECK(!contains(cmd, "--filter syn2_*"),
        "remote cmd does NOT pass the glob unquoted");
  CHECK(contains(cmd, "--filter-class 'must_pass'"),
        "remote cmd forwards --filter-class single-quoted");
  CHECK(contains(cmd, "--filter-category 'hip/*'"),
        "remote cmd forwards --filter-category single-quoted");
  CHECK(contains(cmd, "--hide-na"),
        "remote cmd forwards --hide-na when set");

  // --policy is made ABSOLUTE (starts with '/') and single-quoted.
  CHECK(contains(cmd, "--policy '/"),
        "remote cmd makes --policy absolute and single-quotes it");

  // R3 hardening: values with spaces / shell metacharacters must be single-quoted,
  // not interpolated raw — no word-splitting, no injection into the remote shell.
  {
    const std::string evil = build_remote_command(
        p, 0, 2, "/opt/my tools/sa", "h:1", "acme; rm -rf /");
    CHECK(contains(evil, "'/opt/my tools/sa'"),
          "binary path containing a space is single-quoted");
    CHECK(contains(evil, "--vendor 'acme; rm -rf /'"),
          "vendor with shell metacharacters is single-quoted (no injection)");
    CHECK(!contains(evil, "--vendor acme; rm"),
          "metacharacters never reach the remote shell unquoted");
  }

  // When isolate is NOT explicit, --isolate is omitted entirely.
  {
    RemoteCommandParams q;
    q.kind = "func";
    q.isolate = "all";
    q.isolate_explicit = false;
    const std::string c2 = build_remote_command(q, 1, 2, "/abs/hva",
                                                "n0:1", "synthetic");
    CHECK(!contains(c2, "--isolate"),
          "remote cmd omits --isolate when not explicit");
    CHECK(!contains(c2, "--filter"),
          "remote cmd omits --filter when empty");
    CHECK(!contains(c2, "--filter-class"),
          "remote cmd omits --filter-class when empty");
    CHECK(!contains(c2, "--filter-category"),
          "remote cmd omits --filter-category when empty");
    CHECK(!contains(c2, "--hide-na"),
          "remote cmd omits --hide-na when false");
    CHECK(!contains(c2, "--policy"),
          "remote cmd omits --policy when empty");
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
  RemoteCommandParams p;
  p.kind = "func";

  // Empty env (the default/zero value) -> command identical to before env forwarding:
  // no "env" prefix at all.
  {
    const std::string cmd =
        build_remote_command(p, 0, 1, "/abs/hva", "n0:1", "synthetic");
    CHECK(cmd.rfind("'/abs/hva'", 0) == 0,
          "empty env: command starts with the binary, no env prefix");
    CHECK(!contains(cmd, "env '"), "empty env: no 'env ...' prefix at all");
  }

  // Two vars -> "env 'A=1' 'B=x y' " immediately precedes the binary, in the
  // order given (do_nodes is responsible for sorting before calling in).
  {
    p.env = {{"A", "1"}, {"B", "x y"}};
    const std::string cmd =
        build_remote_command(p, 0, 1, "/abs/hva", "n0:1", "synthetic");
    CHECK(cmd.rfind("env 'A=1' 'B=x y' '/abs/hva'", 0) == 0,
          "two vars: command starts with env 'A=1' 'B=x y' '<binary>'");
  }

  // A value containing a single quote is escaped the canonical POSIX way,
  // and a value containing an embedded newline survives literally (single
  // quotes are the one POSIX quoting form that preserves it).
  {
    p.env = {{"Q", "a'b"}, {"N", "line1\nline2"}};
    const std::string cmd =
        build_remote_command(p, 0, 1, "/abs/hva", "n0:1", "synthetic");
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
  test_build_ssh_argv();
  test_build_remote_command();
  test_build_remote_command_env_forwarding();
  test_forward_env_pattern_matches();
  test_is_valid_forward_env_pattern();
  test_select_forward_env();
  test_parse_forward_env_spec();
  test_split_nodes();
}

}  // namespace accept_test
