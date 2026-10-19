#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# selftest.sh — end-to-end self-test of the acceptance harness against the
# synthetic oracle ("Synthetic validation matrix").
#
# This is the Step-4 validation deliverable. It drives the real `hva`
# binary over the synthetic vendor and asserts, via python3 parsing of
# report.json, that the full gate matrix, the synth fields, discovery mode, the
# explicit-missing-policy hard error, and the --filter masking fix all behave.
#
# Usage:  selftest.sh [path-to-hva-binary] [path-to-source-dir]
#   binary default:     ./build/hva
#   source-dir default: .                (so policy/synthetic.csv is found)
#
# Exits non-zero if ANY check fails. No `set -e` (the full gate run is EXPECTED
# to exit 1 — we capture that code explicitly rather than aborting on it).

set -u

# ---------------------------------------------------------------------------
# Arg + path resolution (make everything absolute so cwd changes are safe).
# ---------------------------------------------------------------------------
abspath() {
  # Resolve a path to absolute without requiring it to exist as a dir we cd into.
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s/%s\n' "$(pwd)" "$1" ;;
  esac
}

BIN_ARG="${1:-./build/hva}"
SRC_ARG="${2:-.}"

BIN="$(abspath "$BIN_ARG")"
SRC="$(cd "$SRC_ARG" 2>/dev/null && pwd)"
if [ -z "${SRC:-}" ]; then
  printf 'selftest: source dir not found: %s\n' "$SRC_ARG" >&2
  exit 2
fi
# --policy now names a DIRECTORY containing per-vendor CSVs (policy/<vendor>.csv).
POLICY="$SRC/policy"

if [ ! -x "$BIN" ]; then
  printf 'selftest: hva binary not found or not executable: %s\n' "$BIN" >&2
  exit 2
fi
if [ ! -f "$POLICY/synthetic.csv" ]; then
  printf 'selftest: policy/synthetic.csv not found at: %s\n' "$POLICY/synthetic.csv" >&2
  exit 2
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  printf 'selftest: python3 not found (set PYTHON=...)\n' >&2
  exit 2
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/hva_selftest.XXXXXX")"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

CHECKS=0
FAILURES=0

pass() { CHECKS=$((CHECKS + 1)); printf 'PASS: %s\n' "$1"; }
fail() {
  CHECKS=$((CHECKS + 1))
  FAILURES=$((FAILURES + 1))
  printf 'FAIL: %s\n' "$1"
  if [ -n "${2:-}" ]; then
    printf '      %s\n' "$2"
  fi
}

# Run the binary with args, capturing exit code into RC and output into a log.
# Never aborts the script on non-zero exit (we assert on RC explicitly).
RC=0
run_bin() {
  local log="$1"; shift
  "$BIN" "$@" >"$log" 2>&1
  RC=$?
  return 0
}

# ---------------------------------------------------------------------------
# The oracle table (full unfiltered gate run). Each line:
#   id class status gate_fail flagged
# ---------------------------------------------------------------------------
ORACLE="$TMP/oracle.txt"
cat >"$ORACLE" <<'EOF'
syn_pass must_pass pass false false
syn_fail must_pass fail true false
syn_skip must_pass fail true false
syn_crash_abort must_pass fail true false
syn_crash_segv must_pass fail true false
syn_hang must_pass fail true false
syn_hang_ignore_sigterm must_pass fail true false
syn_exit_nonzero must_pass fail true false
syn_corrupt_result must_pass fail true false
syn_spawn_grandchild_hang must_pass fail true false
syn_known_failure_fail known_failure fail true false
syn_known_failure_pass known_failure fail true false
syn_na na na false false
EOF
# syn_unclassified is intentionally absent from synthetic.csv -> not run.

# Expected summary counts (known_failure outputs as fail).
EXP_PASS=1; EXP_FAIL=11; EXP_SKIP=0; EXP_NA=1; EXP_KB=0; EXP_UNC=0

# ===========================================================================
# Check (a): full gate matrix.
# ===========================================================================
REP_A="$TMP/rep_full.json"
LOG_A="$TMP/full.log"
run_bin "$LOG_A" --vendor synthetic --policy "$POLICY/synthetic.csv" --report "$REP_A"
RC_A=$RC

if [ "$RC_A" -eq 1 ]; then
  pass "full gate run exits 1 (expected gate fail)"
else
  fail "full gate run exit code" "expected 1, got $RC_A"
  cat "$LOG_A"
fi

if [ ! -f "$REP_A" ]; then
  fail "full gate report.json written" "missing: $REP_A"
  cat "$LOG_A"
else
  # Hand the report + oracle to python3 for structural assertions.
  "$PY" - "$REP_A" "$ORACLE" "$EXP_PASS" "$EXP_FAIL" "$EXP_SKIP" "$EXP_NA" \
      "$EXP_KB" "$EXP_UNC" <<'PYEOF' >"$TMP/a_out.txt" 2>&1
import json, sys
rep_path, oracle_path = sys.argv[1], sys.argv[2]
exp = list(map(int, sys.argv[3:9]))  # pass fail skip na kb unc
errs = []
with open(rep_path) as f:
    rep = json.load(f)

if rep.get("mode") != "gate":
    errs.append('mode != "gate": %r' % rep.get("mode"))
if rep.get("filtered") is not False:
    errs.append('filtered != false: %r' % rep.get("filtered"))

s = rep.get("summary", {})
got = [s.get("passed"), s.get("failed"), s.get("skipped"), s.get("not_applicable")]
names = ["passed", "failed", "skipped", "not_applicable"]
exp_counts = exp[:4]  # pass fail skip na (known_failure/unclassified gone)
for n, g, e in zip(names, got, exp_counts):
    if g != e:
        errs.append("summary.%s = %r, expected %d" % (n, g, e))

tests = {t["id"]: t for t in rep.get("tests", [])}
oracle_ids = set()
with open(oracle_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tid, cls, status, gate, flag = line.split()
        oracle_ids.add(tid)
        if tid not in tests:
            errs.append("missing test row: %s" % tid)
            continue
        t = tests[tid]
        if t.get("class") != cls:
            errs.append("%s class=%r expected %s" % (tid, t.get("class"), cls))
        if t.get("status") != status:
            errs.append("%s status=%r expected %s" % (tid, t.get("status"), status))
        if t.get("gate_fail") != (gate == "true"):
            errs.append("%s gate_fail=%r expected %s" % (tid, t.get("gate_fail"), gate))
        if t.get("flagged") != (flag == "true"):
            errs.append("%s flagged=%r expected %s" % (tid, t.get("flagged"), flag))

# Every report row must be in the oracle (no extras).
extra = set(tests) - oracle_ids
if extra:
    errs.append("unexpected test rows: %s" % sorted(extra))

# A runtime failure must keep its own reason in the JSON artifact (must_pass has
# no policy reason, so the field would be null if the runtime reason were dropped).
sf = tests.get("syn_fail")
if sf is not None and not sf.get("reason"):
    errs.append("syn_fail.reason is empty/null (runtime fail reason dropped)")

# Single-node report still emits a nodes[] entry (rank 0).
nodes = rep.get("nodes", [])
if not nodes or nodes[0].get("rank") != 0:
    errs.append("nodes[] missing rank 0: %r" % nodes)

if errs:
    print("\n".join(errs))
    sys.exit(1)
sys.exit(0)
PYEOF
  if [ $? -eq 0 ]; then
    pass "full gate matrix matches oracle (mode/filtered/summary/all rows)"
  else
    fail "full gate matrix matches oracle" "$(cat "$TMP/a_out.txt")"
  fi
fi

# ===========================================================================
# Check (b): synth fields (crash/timeout/corrupt) on the report from (a).
# ===========================================================================
if [ -f "$REP_A" ]; then
  "$PY" - "$REP_A" <<'PYEOF' >"$TMP/b_out.txt" 2>&1
import json, sys
with open(sys.argv[1]) as f:
    rep = json.load(f)
tests = {t["id"]: t for t in rep.get("tests", [])}
want = {
    "syn_crash_abort": "crash",
    "syn_crash_segv": "crash",
    "syn_exit_nonzero": "crash",
    "syn_hang": "timeout",
    "syn_hang_ignore_sigterm": "timeout",
    "syn_corrupt_result": "corrupt",
}
errs = []
for tid, exp in want.items():
    t = tests.get(tid)
    if t is None:
        errs.append("missing %s" % tid)
        continue
    got = t.get("synth")
    if got != exp:
        errs.append("%s synth=%r expected %s" % (tid, got, exp))
if errs:
    print("\n".join(errs)); sys.exit(1)
sys.exit(0)
PYEOF
  if [ $? -eq 0 ]; then
    pass "synth fields correct (crash/timeout/corrupt)"
  else
    fail "synth fields correct" "$(cat "$TMP/b_out.txt")"
  fi
else
  fail "synth fields correct" "no report from check (a)"
fi

# ===========================================================================
# Check (c): no policy file -> WARN on stderr, exit 1 (gate active,
# all tests run as MUST_PASS; crash/hang tests gate-fail), report written.
# ===========================================================================
DISCO_DIR="$TMP/disco"   # a dir whose ./policy/ lacks synthetic.csv
mkdir -p "$DISCO_DIR/policy"   # policy dir present but no <vendor>.csv
REP_C="$TMP/rep_disco.json"
LOG_C="$TMP/disco.log"
# Run from a dir whose ./policy/ lacks synthetic.csv and do NOT pass --policy.
( cd "$DISCO_DIR" && "$BIN" --vendor synthetic --report "$REP_C" >"$LOG_C" 2>&1 )
RC_C=$?

if [ "$RC_C" -eq 1 ] && [ -f "$REP_C" ]; then
  # Verify WARN was printed to stderr and mode is "discovery".
  "$PY" - "$REP_C" "$LOG_C" <<'PYEOF' >"$TMP/c_out.txt" 2>&1
import json, sys
with open(sys.argv[1]) as f:
    rep = json.load(f)
with open(sys.argv[2]) as f:
    log = f.read()
errs = []
if rep.get("mode") != "discovery":
    errs.append('mode != "discovery": %r' % rep.get("mode"))
if "WARN:" not in log and "no policy file found" not in log:
    errs.append("no WARN message in output (WARN expected when no policy)")
if errs:
    print("\n".join(errs)); sys.exit(1)
sys.exit(0)
PYEOF
  if [ $? -eq 0 ]; then
    pass "discovery mode (exit 1, WARN printed, mode=discovery, gate active)"
  else
    fail "discovery mode" "$(cat "$TMP/c_out.txt")"
  fi
else
  fail "discovery mode" "exit=$RC_C report_exists=$( [ -f "$REP_C" ] && echo yes || echo no )"
  cat "$LOG_C"
fi

# ===========================================================================
# Check (d): explicit missing policy -> nonzero exit, no report written.
# ===========================================================================
REP_D="$TMP/rep_missing.json"
LOG_D="$TMP/missing.log"
rm -f "$REP_D"
run_bin "$LOG_D" --vendor synthetic --policy /no/such/policy_file.csv --report "$REP_D"
RC_D=$RC
if [ "$RC_D" -ne 0 ] && [ ! -f "$REP_D" ]; then
  pass "explicit missing policy (nonzero exit, no report)"
else
  fail "explicit missing policy" \
       "exit=$RC_D report_exists=$( [ -f "$REP_D" ] && echo yes || echo no )"
  cat "$LOG_D"
fi

# ===========================================================================
# Check (e): --filter masking fix. Filter to syn_pass only -> report contains
# ONLY syn_pass, filtered==true, syn_fail ABSENT (not present as na), and the run
# does NOT gate-fail purely from omission.
# ===========================================================================
REP_E="$TMP/rep_filter.json"
LOG_E="$TMP/filter.log"
run_bin "$LOG_E" --filter syn_pass --vendor synthetic --policy "$POLICY/synthetic.csv" \
        --report "$REP_E"
RC_E=$RC

if [ ! -f "$REP_E" ]; then
  fail "filter masking fix" "no report written: $REP_E"
  cat "$LOG_E"
else
  "$PY" - "$REP_E" "$RC_E" <<'PYEOF' >"$TMP/e_out.txt" 2>&1
import json, sys
with open(sys.argv[1]) as f:
    rep = json.load(f)
rc = int(sys.argv[2])
errs = []
ids = [t["id"] for t in rep.get("tests", [])]
if ids != ["syn_pass"]:
    errs.append("tests[] ids = %r, expected exactly ['syn_pass']" % ids)
if rep.get("filtered") is not True:
    errs.append('filtered != true: %r' % rep.get("filtered"))
if any(t["id"] == "syn_fail" for t in rep.get("tests", [])):
    errs.append("syn_fail present in filtered report (masking bug)")
# syn_pass passes, so the filtered run must NOT gate-fail (exit 0). The point of
# the fix: omission of failing tests must not be the thing that flips the gate.
if rc != 0:
    errs.append("filtered run gate-failed (exit %d); syn_pass-only should pass" % rc)
if errs:
    print("\n".join(errs)); sys.exit(1)
sys.exit(0)
PYEOF
  if [ $? -eq 0 ]; then
    pass "filter masking fix (only syn_pass, filtered=true, syn_fail absent, no spurious gate fail)"
  else
    fail "filter masking fix" "$(cat "$TMP/e_out.txt")"
    cat "$LOG_E"
  fi
fi

# ===========================================================================
# Check (f): 2-rank smoke (Step 5). Spawn two local worker processes that
# rendezvous over TCP on localhost, run the TwoNode syn2_* tests in-process with
# the shared SocketBootstrap, and reconcile the per-rank results into one report.
# Asserts:
#   * syn2_barrier -> pass (proves real cross-process barrier+exchange);
#   * syn2_r0fail_r1pass -> fail, gate_fail true, reduced_from {0:fail,1:pass};
#   * syn2_r0pass_r1passive -> pass, reduced_from {0:pass,1:none};
#   * overall gate FAIL (exit 1) because syn2_r0fail_r1pass fails.
# ===========================================================================
REP_F="$TMP/rep2.json"
LOG_F="$TMP/twonode.log"
# 'syn2_[!h]*' excludes syn2_hang: that test is ACCEPT_SYN_HANG-gated (check
# (l) below exercises it on its own, with the env var set) and would
# otherwise show up here as an unrelated MUST_PASS/skip gate-fail row since
# ACCEPT_SYN_HANG is not set in this invocation.
run_bin "$LOG_F" --local-ranks 2 --filter 'syn2_[!h]*' --vendor synthetic \
        --policy "$POLICY/synthetic.csv" --report "$REP_F"
RC_F=$RC

if [ ! -f "$REP_F" ]; then
  fail "2-rank smoke" "no report written: $REP_F"
  cat "$LOG_F"
else
  "$PY" - "$REP_F" "$RC_F" <<'PYEOF' >"$TMP/f_out.txt" 2>&1
import json, sys
with open(sys.argv[1]) as f:
    rep = json.load(f)
rc = int(sys.argv[2])
errs = []
tests = {t["id"]: t for t in rep.get("tests", [])}

def need(tid):
    t = tests.get(tid)
    if t is None:
        errs.append("missing test row: %s" % tid)
    return t

t = need("syn2_barrier")
if t is not None:
    if t.get("status") != "pass":
        errs.append("syn2_barrier status=%r expected pass "
                    "(cross-process exchange failed)" % t.get("status"))
    rf = t.get("reduced_from", {})
    if rf.get("0") != "pass" or rf.get("1") != "pass":
        errs.append("syn2_barrier reduced_from=%r expected {0:pass,1:pass}" % rf)

t = need("syn2_r0fail_r1pass")
if t is not None:
    if t.get("status") != "fail":
        errs.append("syn2_r0fail_r1pass status=%r expected fail" % t.get("status"))
    if t.get("gate_fail") is not True:
        errs.append("syn2_r0fail_r1pass gate_fail=%r expected true" % t.get("gate_fail"))
    rf = t.get("reduced_from", {})
    if rf.get("0") != "fail" or rf.get("1") != "pass":
        errs.append("syn2_r0fail_r1pass reduced_from=%r expected {0:fail,1:pass}" % rf)

t = need("syn2_r0pass_r1passive")
if t is not None:
    if t.get("status") != "pass":
        errs.append("syn2_r0pass_r1passive status=%r expected pass "
                    "(none is reduction identity)" % t.get("status"))
    rf = t.get("reduced_from", {})
    if rf.get("0") != "pass" or rf.get("1") != "none":
        errs.append("syn2_r0pass_r1passive reduced_from=%r expected {0:pass,1:none}" % rf)

if rc != 1:
    errs.append("2-rank gate exit=%d expected 1 (syn2_r0fail_r1pass fails)" % rc)

# nodes[] must reflect BOTH ranks, not a single hard-coded localhost/rank-0 entry.
ranks = sorted(n.get("rank") for n in rep.get("nodes", []))
if ranks != [0, 1]:
    errs.append("nodes[] ranks=%r expected [0, 1]" % ranks)

if errs:
    print("\n".join(errs)); sys.exit(1)
sys.exit(0)
PYEOF
  if [ $? -eq 0 ]; then
    pass "2-rank smoke (SocketBootstrap barrier+exchange, reconciliation, reduced_from, gate FAIL exit 1)"
  else
    fail "2-rank smoke" "$(cat "$TMP/f_out.txt")"
    cat "$LOG_F"
  fi
fi

# ===========================================================================
# Check (g): launcher bounded-failure regression (CI-runnable, no second host).
# An unreachable node list must fail FAST with exit 4 (deploy/launch failure, not
# a gate verdict) and write NO report. The `.invalid` TLD guarantees an immediate
# NXDOMAIN, so this never blocks on a real network timeout. We still wrap the run
# in `timeout` and fail the check if it trips (proves "does not hang").
# ===========================================================================
REP_G="$TMP/rep_unreach.json"
LOG_G="$TMP/unreach.log"
rm -f "$REP_G"
# Hard wall: 30s. With .invalid the resolve fails instantly; --launch-timeout is
# a belt-and-suspenders bound on the overall ssh deadline.
timeout 30 "$BIN" --nodes nosuchhost1.invalid,nosuchhost2.invalid \
    --vendor synthetic --policy "$POLICY/synthetic.csv" --report "$REP_G" \
    --launch-timeout 20 >"$LOG_G" 2>&1
RC_G=$?

if [ "$RC_G" -eq 124 ]; then
  fail "launcher bounded-unreachable" \
       "run HUNG (timeout tripped at 30s); expected fast exit 4"
  cat "$LOG_G"
elif [ "$RC_G" -eq 4 ] && [ ! -f "$REP_G" ]; then
  pass "launcher bounded-unreachable (exit 4, fast, no report.json)"
else
  fail "launcher bounded-unreachable" \
       "exit=$RC_G (expected 4) report_exists=$( [ -f "$REP_G" ] && echo yes || echo no )"
  cat "$LOG_G"
fi

# ===========================================================================
# Check (h): process-GROUP kill reaches a grandchild (Runner setpgid + killpg).
# Run a focused isolated worker over just syn_spawn_grandchild_hang with
# ACCEPT_GC_PIDFILE set. The test forks a grandchild that records its OWN pid in
# that file and hangs; the test then hangs too. The Runner's watchdog must
# SIGTERM->SIGKILL the WHOLE group, killing the grandchild as well. Assert:
#   * the test row is fail / synth=timeout (the crash/hang gate signal), AND
#   * the recorded grandchild pid is no longer alive (kill -0 fails) — proving
#     the group kill reached it (a bare kill(child_pid) would have leaked it).
# ===========================================================================
GC_PIDFILE="$TMP/gc.pid"
GC_OUT="$TMP/gc_results.kv"
LOG_H="$TMP/gc.log"
rm -f "$GC_PIDFILE" "$GC_OUT"
# Bound with `timeout` so a regression that leaves the worker hung fails the
# check instead of hanging the suite. The test timeout is 2s; give generous slack.
# --isolate all is REQUIRED: this is a single-test run (one --filter match), which
# would otherwise default to in-process ("none") and the hang would not be caught
# by the watchdog. We are specifically exercising the isolated group-kill path.
ACCEPT_GC_PIDFILE="$GC_PIDFILE" timeout 30 "$BIN" --worker --rank 0 --size 1 \
    --results-out "$GC_OUT" --filter syn_spawn_grandchild_hang --isolate all \
    --vendor synthetic --policy "$POLICY/synthetic.csv" >"$LOG_H" 2>&1
RC_H=$?

if [ "$RC_H" -eq 124 ]; then
  fail "grandchild group-kill" "worker HUNG (timeout tripped); group kill failed?"
  cat "$LOG_H"
elif [ ! -f "$GC_OUT" ]; then
  fail "grandchild group-kill" "no worker results file: $GC_OUT (exit=$RC_H)"
  cat "$LOG_H"
else
  # 1) The test row must be fail / synth=timeout.
  ROW_OK=1
  if ! grep -q '^id=syn_spawn_grandchild_hang$' "$GC_OUT"; then ROW_OK=0; fi
  if ! grep -q '^status=fail$' "$GC_OUT"; then ROW_OK=0; fi
  if ! grep -q '^synth=timeout$' "$GC_OUT"; then ROW_OK=0; fi

  # 2) The grandchild pid must be recorded and now DEAD. Poll briefly to let the
  #    SIGKILL'd grandchild be reaped by init.
  GC_PID=""
  if [ -f "$GC_PIDFILE" ]; then GC_PID="$(head -n1 "$GC_PIDFILE" | tr -dc '0-9')"; fi
  GC_DEAD=0
  if [ -n "$GC_PID" ]; then
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if kill -0 "$GC_PID" 2>/dev/null; then
        sleep 0.2
      else
        GC_DEAD=1
        break
      fi
    done
  fi

  if [ "$ROW_OK" -eq 1 ] && [ -n "$GC_PID" ] && [ "$GC_DEAD" -eq 1 ]; then
    pass "grandchild group-kill (row fail/timeout; grandchild pid $GC_PID is dead)"
  else
    fail "grandchild group-kill" \
         "row_ok=$ROW_OK gc_pid='$GC_PID' gc_dead=$GC_DEAD (still alive => leak)"
    echo "--- results ---"; cat "$GC_OUT"
    echo "--- log ---"; cat "$LOG_H"
    # Best-effort: don't leave a leaked grandchild running across the suite.
    [ -n "$GC_PID" ] && kill -9 "$GC_PID" 2>/dev/null
  fi
fi

# ===========================================================================
# Check (i): watchdog timeout exit_code accuracy. A timed-out test that HONORS
# SIGTERM in the grace window must record exit_code -15 (SIGTERM); one that
# ignores SIGTERM and is force-killed must record -9 (SIGKILL). Both stay
# synth=timeout. Guards against regressing to a hard-coded -SIGKILL for every
# timeout (which would erase the SIGTERM-vs-SIGKILL watchdog signal).
# ===========================================================================
TO_OUT="$TMP/timeout.kv"
LOG_I="$TMP/timeout.log"
rm -f "$TO_OUT"
timeout 30 "$BIN" --worker --rank 0 --size 1 --results-out "$TO_OUT" \
    --filter 'syn_hang*' --kind all --vendor synthetic --policy "$POLICY/synthetic.csv" \
    >"$LOG_I" 2>&1
RC_I=$?
ec_of() {  # exit_code for the @test block whose id == $1 (exact match)
  awk -v id="$1" '
    /^id=/        { cur = substr($0, 4) }
    /^exit_code=/ { if (cur == id) { print substr($0, 11); exit } }
  ' "$TO_OUT"
}
if [ "$RC_I" -eq 124 ] || [ ! -f "$TO_OUT" ]; then
  fail "timeout exit_code accuracy" "worker hung or no results (exit=$RC_I)"
  cat "$LOG_I"
else
  EC_TERM="$(ec_of syn_hang)"
  EC_KILL="$(ec_of syn_hang_ignore_sigterm)"
  if [ "$EC_TERM" = "-15" ] && [ "$EC_KILL" = "-9" ]; then
    pass "timeout exit_code accuracy (SIGTERM-honoring -> -15, force-killed -> -9)"
  else
    fail "timeout exit_code accuracy" \
         "syn_hang='$EC_TERM' (want -15), syn_hang_ignore_sigterm='$EC_KILL' (want -9)"
    cat "$TO_OUT"
  fi
fi

# ===========================================================================
# Check (j): HW/FW discovery + environment descriptor (M1a/M1b). With
# ACCEPT_FAKE_ENV=1 the probe returns a deterministic descriptor (no hardware
# needed); the workers inherit the env var across fork/execv. Run a 2-rank
# localhost launch and assert every nodes[].environment is the fake descriptor
# and the run is homogeneous.
# ===========================================================================
REP_J="$TMP/rep_env.json"
LOG_J="$TMP/env.log"
ACCEPT_FAKE_ENV=1 timeout 60 "$BIN" --local-ranks 2 --filter 'syn2_*' \
    --vendor synthetic --policy "$POLICY/synthetic.csv" --report "$REP_J" >"$LOG_J" 2>&1
RC_J=$?

if [ "$RC_J" -eq 124 ]; then
  fail "environment descriptor (fake env)" "run HUNG (timeout tripped)"
  cat "$LOG_J"
elif [ ! -f "$REP_J" ]; then
  fail "environment descriptor (fake env)" "no report written: $REP_J"
  cat "$LOG_J"
else
  "$PY" - "$REP_J" <<'PYEOF' >"$TMP/j_out.txt" 2>&1
import json, sys
with open(sys.argv[1]) as f:
    rep = json.load(f)
errs = []
if rep.get("homogeneous") is not True:
    errs.append('homogeneous != true: %r' % rep.get("homogeneous"))
nodes = rep.get("nodes", [])
if len(nodes) != 2:
    errs.append("expected 2 nodes, got %d" % len(nodes))
for n in nodes:
    env = n.get("environment", {})
    if env.get("vendor") != "thor":
        errs.append("rank %r environment.vendor=%r expected 'thor'"
                    % (n.get("rank"), env.get("vendor")))
    if not env.get("rocm_version"):
        errs.append("rank %r environment.rocm_version empty/null"
                    % n.get("rank"))
    if env.get("gpu_count") != 8:
        errs.append("rank %r environment.gpu_count=%r expected 8"
                    % (n.get("rank"), env.get("gpu_count")))
    rdma = env.get("rdma_devices", [])
    if not rdma or rdma[0].get("device") != "bnxt_re0":
        errs.append("rank %r rdma_devices missing bnxt_re0: %r"
                    % (n.get("rank"), rdma))
    # M1c: symbols map present with the fake-env hsa probe true.
    syms = env.get("symbols")
    if not isinstance(syms, dict):
        errs.append("rank %r environment.symbols missing/!dict: %r"
                    % (n.get("rank"), syms))
    elif syms.get("hsa_amd_portable_export_dmabuf") is not True:
        errs.append("rank %r symbols.hsa_amd_portable_export_dmabuf=%r expected true"
                    % (n.get("rank"), syms.get("hsa_amd_portable_export_dmabuf")))
    # M1c: gpu_model / driver_version match the fake-env seam exactly (a mere
    # non-empty check would pass on any string and miss serialization drift).
    if env.get("gpu_model") != "gfx-fake":
        errs.append("rank %r environment.gpu_model=%r expected 'gfx-fake'"
                    % (n.get("rank"), env.get("gpu_model")))
    if env.get("driver_version") != "0.0-fake":
        errs.append("rank %r environment.driver_version=%r expected '0.0-fake'"
                    % (n.get("rank"), env.get("driver_version")))
if errs:
    print("\n".join(errs)); sys.exit(1)
sys.exit(0)
PYEOF
  if [ $? -eq 0 ]; then
    pass "environment descriptor (fake env: vendor=thor, rocm non-empty, gpu_count=8, symbols/gpu_model/driver_version, homogeneous)"
  else
    fail "environment descriptor (fake env)" "$(cat "$TMP/j_out.txt")"
    cat "$LOG_J"
  fi
fi

# ===========================================================================
# Check (k): `when` predicate engine with ACCEPT_FAKE_ENV=1.
# Uses policy/synthetic_when.csv which has rows with `when` predicates
# calibrated to the deterministic fake env (rocm_version="0.0-fake",
# driver_version="0.0-fake", gpu_count=8, libionic absent,
# hsa_amd_portable_export_dmabuf present).
#
# Asserts per the synthetic_when.csv comments:
#   syn_pass  -> na  (rocm<0.1 fires: 0.0-fake < 0.1 is true)
#   syn_fail  -> na  (single-row MUST_PASS,rocm>=1.0 fails: 0.0 not >= 1.0; no catch-all -> NA)
#   syn_skip  -> na  (rocm<0.1 fires, piecewise first row)
#   syn_crash_abort -> na (multi-term: rocm<0.1 AND gpu>=8 both true)
#   syn_crash_segv  -> na (sym:hsa_amd_portable_export_dmabuf present in fake env)
#   syn_hang        -> na (!sym:libionic: libionic absent in fake env)
# Also checks that syn_pass.when_matched == "rocm<0.1".
# ===========================================================================
if [ ! -f "$POLICY/synthetic_when.csv" ]; then
  fail "when-predicate (fake env)" \
       "policy/synthetic_when.csv not found: $POLICY/synthetic_when.csv"
else
  REP_K="$TMP/rep_when.json"
  LOG_K="$TMP/when.log"
  ACCEPT_FAKE_ENV=1 timeout 60 "$BIN" --vendor synthetic_when \
      --policy "$POLICY/synthetic_when.csv" --report "$REP_K" >"$LOG_K" 2>&1
  RC_K=$?

  if [ "$RC_K" -eq 124 ]; then
    fail "when-predicate (fake env)" "run HUNG (timeout tripped)"
    cat "$LOG_K"
  elif [ ! -f "$REP_K" ]; then
    fail "when-predicate (fake env)" "no report written: $REP_K"
    cat "$LOG_K"
  else
    "$PY" - "$REP_K" "$RC_K" <<'PYEOF' >"$TMP/k_out.txt" 2>&1
import json, sys
with open(sys.argv[1]) as f:
    rep = json.load(f)
rc = int(sys.argv[2])
errs = []
tests = {t["id"]: t for t in rep.get("tests", [])}

def check_status(tid, want_status, want_class, label):
    t = tests.get(tid)
    if t is None:
        errs.append("missing test row: %s (%s)" % (tid, label))
        return
    if t.get("status") != want_status:
        errs.append("%s status=%r expected %s (%s)" % (tid, t.get("status"), want_status, label))
    if t.get("class") != want_class:
        errs.append("%s class=%r expected %s (%s)" % (tid, t.get("class"), want_class, label))

# syn_pass: rocm<0.1 fires on fake env (0.0-fake < 0.1) -> NA
check_status("syn_pass", "na", "na", "rocm<0.1 fires")

# syn_fail: single-row MUST_PASS,rocm>=1.0. Fake env has rocm=0.0-fake (<1.0),
# predicate fails, rocm is a software dim -> SKIP.
check_status("syn_fail", "skip", "skip", "single-row: rocm>=1.0 fails -> SKIP (software dim)")

# syn_skip: rocm<0.1 fires -> NA (piecewise first row)
check_status("syn_skip", "na", "na", "piecewise rocm<0.1 fires")

# syn_crash_abort: multi-term AND rocm<0.1;gpu>=8 both hold -> NA
check_status("syn_crash_abort", "na", "na", "multi-term AND both true")

# syn_crash_segv: sym:hsa_amd_portable_export_dmabuf present -> NA
check_status("syn_crash_segv", "na", "na", "sym present in fake env")

# syn_hang: !sym:libionic (libionic absent) -> NA
check_status("syn_hang", "na", "na", "!sym:libionic absent")

# when_matched: syn_pass was selected by "rocm<0.1"
t_pass = tests.get("syn_pass")
if t_pass is not None:
    wm = t_pass.get("when_matched")
    if wm != "rocm<0.1":
        errs.append("syn_pass when_matched=%r expected 'rocm<0.1'" % wm)

# syn_fail: rocm>=1.0 fails on fake env; rocm is a software dim -> SKIP.
t_fail = tests.get("syn_fail")
if t_fail is None:
    errs.append("syn_fail row missing from report")
elif t_fail.get("status") != "skip":
    errs.append("syn_fail status=%r expected skip (single-row: rocm>=1.0 fails -> SKIP)" % t_fail.get("status"))

if errs:
    print("\n".join(errs)); sys.exit(1)
sys.exit(0)
PYEOF
    if [ $? -eq 0 ]; then
      pass "when-predicate (fake env: na/must_pass routing, multi-term AND, sym, !sym, when_matched)"
    else
      fail "when-predicate (fake env)" "$(cat "$TMP/k_out.txt")"
      cat "$LOG_K"
    fi
  fi
fi

# ===========================================================================
# Check (l): worker deadline watchdog. syn2_hang (ACCEPT_SYN_HANG=1)
# makes both --local-ranks workers spin forever in-process (a TwoNode test
# has no per-test timeout, see Runner::run_in_process). --launch-timeout 3
# gives each worker --deadline-sec 13 (launch_timeout+10); its own watchdog
# thread must SIGKILL the worker's process group well before the launcher's
# own 90s overall-wait bound, so the launcher observes an abnormal child exit
# and returns 4. Bounded with `timeout` so a regression here fails the check
# instead of hanging the whole selftest; the pgrep below is scoped to THIS
# worktree's binary path (not a bare 'hva --worker') because other agents run
# their own hva builds concurrently on this host.
# ===========================================================================
LOG_L="$TMP/hang.log"
timeout 30 env ACCEPT_SYN_HANG=1 "$BIN" --local-ranks 2 --filter syn2_hang \
    --launch-timeout 3 --vendor synthetic --policy "$POLICY/synthetic.csv" \
    >"$LOG_L" 2>&1
RC_L=$?

# Scoped stray-process check. do_local_ranks execv's "/proc/self/exe" with
# that LITERAL string as argv[0] (so the worker's cmdline does NOT contain
# $BIN's path -- `pgrep -f "$BIN --worker"` would never match). Instead:
# shortlist candidates with a plain '--worker' cmdline match, then keep only
# the ones whose /proc/<pid>/exe resolves to THIS worktree's built binary.
# That is the scoping this host actually needs: other agents build and run
# their OWN hva binaries concurrently (different real path => filtered out),
# so this never mistakes their workers for a leak from this run.
BIN_REAL="$(readlink -f "$BIN" 2>/dev/null || printf '%s' "$BIN")"
stray_workers() {
  local pid exe_real out=""
  for pid in $(pgrep -f -- '--worker' 2>/dev/null); do
    exe_real="$(readlink -f "/proc/$pid/exe" 2>/dev/null)" || continue
    [ "$exe_real" = "$BIN_REAL" ] && out="$out $pid"
  done
  printf '%s\n' "$out" | xargs -n1 2>/dev/null
}

if [ "$RC_L" -eq 124 ]; then
  fail "worker deadline watchdog" \
       "run HUNG (timeout tripped at 30s); watchdog did not fire"
  cat "$LOG_L"
  # Best-effort: do not leave the hang running for the rest of the suite.
  STRAY="$(stray_workers)"
  [ -n "$STRAY" ] && kill -9 $STRAY 2>/dev/null
elif [ "$RC_L" -ne 4 ]; then
  fail "worker deadline watchdog" "exit=$RC_L expected 4 (abnormal worker exit)"
  cat "$LOG_L"
  STRAY="$(stray_workers)"
  [ -n "$STRAY" ] && kill -9 $STRAY 2>/dev/null
else
  # Give init a moment to reap the SIGKILL'd workers before the stray check.
  STRAY=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    STRAY="$(stray_workers)"
    [ -z "$STRAY" ] && break
    sleep 0.3
  done
  if [ -z "$STRAY" ]; then
    pass "worker deadline watchdog (exit 4, no stray hva --worker processes)"
  else
    fail "worker deadline watchdog" "stray worker process(es) remain: $STRAY"
    kill -9 $STRAY 2>/dev/null
  fi
fi

# ---------------------------------------------------------------------------
# Final verdict.
# ---------------------------------------------------------------------------
printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf 'SELFTEST PASSED (%d checks, %d failures)\n' "$CHECKS" "$FAILURES"
  exit 0
else
  printf 'SELFTEST FAILED (%d checks, %d failures)\n' "$CHECKS" "$FAILURES"
  exit 1
fi
