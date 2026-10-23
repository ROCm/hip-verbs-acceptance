# HIP/Verbs Acceptance Suite

A standalone acceptance-test harness (no RCCL, no GoogleTest, no third-party JSON)
that runs HIP and RDMA/Verbs primitives as a gate before the main test suite and
when validating new hardware, firmware, or a ROCm bump. It answers one question:
**are the low-level GPU + NIC primitives RCCL depends on healthy on this machine?**

New here? Start with the [Quickstart](#quickstart-real-hardware). Bringing the suite
up on a platform for the first time (e.g. MI450)? Read
[docs/NEW_PLATFORM.md](docs/NEW_PLATFORM.md). Adding a test? Read
[docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md).

---

## Quickstart (real hardware)

Prereqs: a C++20 compiler (g++ 11+) and CMake >= 3.20. A ROCm install
(`/opt/rocm`) is required only for the HIP tests (`-DACCEPT_WITH_HIP=ON`) and an
RDMA NIC only for the Verbs tests (`-DACCEPT_WITH_IBVERBS=ON`) — with both flags
off the harness still builds and runs the [synthetic set](#harness-smoke-test-no-gpunic)
(no ROCm needed). Build takes a couple of minutes; runs take seconds to a few
minutes.

### 1. Build

```bash
cd hip-verbs-acceptance
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DACCEPT_WITH_HIP=ON -DACCEPT_WITH_IBVERBS=ON
cmake --build build -j
# produces: build/hva   and   build/unit_tests
```

Drop `-DACCEPT_WITH_HIP=ON` / `-DACCEPT_WITH_IBVERBS=ON` to compile out the HIP or
Verbs tests (e.g. on a box with no GPU or no NIC). With both off you still get the
harness + [synthetic set](#harness-smoke-test-no-gpunic).

To install a packaged copy, `cmake --install build --prefix <dir>` places the
binary at `<dir>/bin/hva` and the policy CSVs at `<dir>/bin/policy/*.csv`,
keeping the auto-policy lookup (`<binary_dir>/policy/...`) working out of the box.

### 2. Run the HIP tests (single node)

```bash
./build/hva --filter-category 'hip/*' --kind func --report report.json
```

### 3. Run the Verbs tests (two ranks / two nodes)

The RDMA data-path tests need two endpoints. The launch flags follow mpirun:
`-n` total ranks, `-ppn` ranks per host, `-hosts` the host list (default
`localhost`; a name or address of this machine runs there without ssh). The
two-rank tests run only with `-n 2`; any other `-n` runs the single-rank tests
on every rank (e.g. `-n 8` covers 8 GPUs at once) and prints a `note:` for the
two-rank tests it leaves out. On one host, use two local ranks:

```bash
./build/hva -n 2 --filter-category 'verbs/*' \
      --kind func --report report.json
```

Each local rank gets its own GPU and NIC: GPU = its local rank (`--device`
overrides), NIC = that GPU's nearest ACTIVE NIC (`--hca` overrides). Both flags
take one value or one per local rank, e.g. `-n 2 --device 0,4` or
`-n 2 --hca bnxt_re0,bnxt_re1`; `--hca` alone picks the GPU nearest to it. When
affinity is unknown (or ties), the first ACTIVE NIC is used and a `note:` line
says why. A `--device` or `--hca` the host does not have is an error (exit 2).
A `topology:` line per rank prints the pair before the test lines
(e.g. `topology: rank 1 on node1: GPU 1 (default), HCA bnxt_re1 (affinity)`),
and each node's `topology` in `report.json` records it. Pinning both ranks to
one NIC (`--hca bnxt_re0`) makes it a single-NIC loopback check.

A GPU-only host (no RDMA device) prints `HCA none` and can run the HIP tests:
`hva --filter-category 'hip/*' --vendor mi350_thor` (or any policy for that
GPU). Without `--vendor`/`--policy` it looks for `policy/<gpu>.csv`, else runs
in discovery mode. The verbs tests skip there, which fails a `MUST_PASS` gate,
so filter them out.

Across two real nodes (rank 0 listens on the first host, rank 1 is launched by
the selected `--remote-launcher`; binary + policy must be at the **same
absolute path** on both, e.g. a shared FS):

```bash
./build/hva -n 2 -hosts host1,host2 --filter-category 'verbs/*' \
      --kind func --report report.json
#   [--remote-launcher ssh|srun] [--port P] [--remote-binary PATH] [--launch-timeout S]
#   ssh only: [--ssh-user U] [--forward-env LIST] [--no-forward-env]
```

Two launchers are available via `--remote-launcher` (default `ssh`):

- **`ssh`** — the default; works anywhere passwordless ssh between the
  allocated nodes is set up (`~/.ssh/config` or `--ssh-user`). Each ssh worker
  otherwise starts with only the remote login environment, not the launcher's,
  so the ssh launcher forwards a fixed set of launcher environment variables to every
  worker; see [Env forwarding](#env-forwarding-ssh-launcher-only) below.
- **`srun`** — use this inside a Slurm allocation where direct ssh between
  compute nodes is blocked (a common cluster policy). It requires
  `SLURM_JOB_ID` to be set (i.e. running from inside `sbatch`/`salloc`) and
  does not accept `--ssh-user` (srun has no login-user concept — it runs as
  the job's submitting user). Needs **Slurm >= 20.11** for the `--overlap`
  flag the launcher passes so the step can share the allocation with the
  batch/interactive step. Unlike ssh, srun propagates the launcher's
  environment to the worker by itself (the launcher passes `--export=ALL`, so
  this holds even if the job set `SLURM_EXPORT_ENV=NONE`), so it does not
  accept `--forward-env` / `--no-forward-env` either.

If the launcher gives up (`--launch-timeout` elapses, or the launcher process
itself dies), every worker it spawned self-destructs on its own
`--deadline-sec` (`launch-timeout + 10`, set automatically -- not a flag you
pass) rather than being left running and holding GPU memory / QPs for the
next job on that node. This only applies to workers started BY the launcher;
running `--worker` by hand (no launcher involved) is unaffected. An older
`--remote-binary` built before this existed does not understand
`--deadline-sec` and exits 2, which fails the run with exit 4 -- rebuild
it with the same hva version as the launcher.

The full `verbs/*` set includes RDMA data-path tests that need two working NIC
endpoints — run those across two real nodes. On a single host, the
`verbs/hip_mixed` subset (GPU memory registration / GDR paths) is a good local
check: `-n 2 --filter-category 'verbs/hip_mixed'`.

Swap `--kind func` for `--kind perf` to also collect latency/bandwidth metrics
into `report.json` (see [Performance](#performance)).

### 4. Read the result

Each test streams one line as it finishes; a summary and the `ACCEPTANCE:` verdict
close the run. Example (HIP tests on an MI300X-class box), showing every status:

```
[  PASS  ] Memcpy_Size_Sweep
[  PASS  ] Device_CanAccessPeer_Matrix
[  PASS  ] Device_P2P_AllPairs_Copy
[   NA   ] Hdp_Flush_Ordering  (gfx<90a only; gfx950 CDNA3+)
[   NA   ] Texture_Object_Lifecycle  (no image hw on CDNA3+ (gfx950))
[  SKIP  ] StreamGetId_Stable  (requires hipStreamGetId (ROCm >= 7.1))
[  FAIL  ] Vmm_Map_NoDeadlock_KernelInFlight  (ROCM-25387 deadlock signature)
report: mode=gate | pass=52 fail=2 skip=5 na=3 | flagged=0 | -> report.json
ACCEPTANCE: FAIL — 2 test(s) failed the gate
```

A clean run (here the `verbs/hip_mixed` subset, `-n 2`) ends:

```
report: mode=gate | pass=17 fail=0 skip=10 na=0 | flagged=0 | -> report.json
ACCEPTANCE: PASS
```

| Status | Meaning | Gate |
|---|---|---|
| `PASS` | test passed | ok |
| `FAIL` | a `MUST_PASS` test failed/was skipped, or a `KNOWN_FAILURE` reproduced | **fails** |
| `SKIP` | software-gated (feature/symbol/ROCm version absent); classified `SKIP` in policy | neutral |
| `NA`   | not applicable on this hardware (permanent); classified `NA` in policy | neutral |

`KNOWN_FAILURE` tests always report `FAIL` on purpose — to keep pressure on the
underlying bug. The gate is green only when nothing `MUST_PASS` fails and
no known bug is present, so `ACCEPTANCE: FAIL` above is the tracked ROCm regression,
not a suite defect.

**Exit codes:** `0` gate pass, `1` gate fail, `2` usage error, `3` I/O error
(cannot open/write a result or report file, mkdtemp/port allocation failure),
`4` launch failure (ssh/srun/unreachable/timeout).

---

## Parameters

| Flag | Purpose |
|---|---|
| `--filter <glob>` | select tests by id (fnmatch), e.g. `'Memcpy_*'` |
| `--filter-category <glob>` | select by category, e.g. `'hip/*'`, `'verbs/*'` |
| `--filter-class <class>` | select by policy class: `must_pass\|known_failure\|skip\|na` |
| `--kind func\|perf\|all` | `func` (default) runs functional tests; `perf`/`all` also record metrics |
| `--vendor <name>` | override NIC vendor auto-detection (picks `policy/<name>.csv`) |
| `--policy <file>` | use a specific policy CSV directly (highest priority) |
| `--report <file>` | where to write `report.json` (default: CWD) |
| `--perf-csv <file>` | also write the measured metrics as CSV (needs `--kind perf\|all`) |
| `-n`, `-np` `<n>` | total ranks (default 1); more than one are launched as workers and reconciled. The two-rank (`verbs/*` two-node) tests run only with `-n 2` |
| `-ppn <p>` | ranks per host (default `ceil(n / #hosts)`), filled in block order as with mpirun |
| `-hosts h1[,h2]` | hosts to place ranks on (default `localhost`); ranks on other hosts start via `--remote-launcher` |
| `--hca <name>[,...]` | RDMA device per local rank (default: the GPU's nearest ACTIVE NIC) |
| `--device <n>[,...]` | GPU (HIP ordinal) per local rank (default: the local rank, or the GPU nearest to `--hca`) |
| `--remote-launcher ssh\|srun` | how a rank on another host is started (default `ssh`; `srun` for Slurm allocations where ssh is blocked) |
| `--forward-env LIST` | ssh launcher only: env vars forwarded to each worker (default list below; `--no-forward-env` disables) |
| `--deadline-sec <s>` | worker-only; set automatically by the launcher (not a flag you pass by hand) so the worker self-destructs if the launcher gives up |
| `--quiet` | print only the summary (suppress per-test lines) |
| `--list` | list all registered tests and exit |
| `--lint` | report registered tests missing from the policy; do not run |

Full list: `./build/hva --help`.

---

## Env forwarding (ssh launcher only)

ssh workers do not automatically share the launcher's environment — the ssh
launcher starts each worker with its **remote login environment**, which can carry
a different (or no) ROCm. To make the common case ("test the ROCm I have
`LD_LIBRARY_PATH`/`ROCM_PATH` pointed at, not whatever is on the node image")
work out of the box, the ssh launcher **forwards a fixed list of variables from the
launcher's own environment by default**:

```
LD_LIBRARY_PATH, ROCM_PATH, PATH, HIP_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES,
HSA_* (prefix match), ACCEPT_* (prefix match)
```

**This is a behaviour change**: before this flag existed, ssh workers always ran
with the remote login environment only. Forwarding is now **ON by default** —
pass `--no-forward-env` to get the old behaviour back. Only variables actually
set in the launcher's own environment are forwarded; the default list forwards
nothing it doesn't find set.

- `--forward-env LIST` replaces the default list with a comma-separated one of
  your own (e.g. `--forward-env 'ROCM_PATH,MY_VAR,ACCEPT_*'`); a trailing `*`
  is a prefix match. `--forward-env ''` forwards nothing (same effect as
  `--no-forward-env`, spelled as an explicit empty list). `--forward-env` and
  `--no-forward-env` together are a usage error (exit 2) — they express
  opposite intents.
- A variable name must match `[A-Za-z_][A-Za-z0-9_]*`, optionally followed by
  one trailing `*`; anything else is a usage error (exit 2). A bare `*` is
  rejected (forwarding "everything" is not an allowed pattern).
- A variable name matching more than one entry (e.g. an explicit `HSA_FOO` and
  the `HSA_*` prefix) is still forwarded exactly once.
- **`PATH` and `--remote-binary`:** the default `--remote-binary` is this
  binary's own resolved `/proc/self/exe` path (always absolute), so forwarding
  `PATH` has no effect on finding it. If you instead pass `--remote-binary` as
  a **bare name** (no `/`) relying on a remote `$PATH` lookup, forwarding
  `PATH` changes which `$PATH` that lookup uses — the launcher's, not the
  node's login shell's. Prefer an absolute `--remote-binary` to avoid that
  dependency entirely.
- Applies to the ssh launcher only. `--remote-launcher srun` always passes the
  launcher's whole environment (`--export=ALL`), so it adds no `env` prefix,
  and `--forward-env` / `--no-forward-env` with srun are a usage error (exit 2)
  rather than silently ignored.

---

## Bringing up a new platform

When there is **no** `policy/<gpu>_<nic>.csv` (nor fallback `policy/<nic>.csv`) for
the detected platform, the suite runs in
**discovery mode**: every test runs and is treated as `MUST_PASS` with the gate
active. That first run tells you what passes, fails, skips, or is not applicable on
the new hardware — the raw material for authoring a policy. The full step-by-step
(classify results, write the policy, re-run gated) is in
[docs/NEW_PLATFORM.md](docs/NEW_PLATFORM.md).

```bash
# no --policy/--vendor: auto-detects the NIC and runs everything as MUST_PASS
./build/hva --filter-category 'hip/*' --kind func --report report.json
```

---

## Adding a test

The suite has a shared test skeleton for both HIP and Verbs. See
[docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md) for the template (including how to
wire in perf from the start) and where to register the test in each policy.

---

## Harness smoke test (no GPU/NIC)

The synthetic set exercises every harness behavior — pass/fail/skip/crash/hang/
timeout/corrupt/known-failure/NA and 2-node reconciliation — with no RDMA or GPU,
so it runs anywhere in seconds. Use it to confirm the harness itself is healthy:

```bash
./build/hva --vendor synthetic --policy policy/synthetic.csv \
      --filter 'syn_*' --report report.json
```

Expected: a per-test line then `ACCEPTANCE: FAIL` (exit 1) — **on purpose**, the
set deliberately contains failing/crash/hang/known-failure fixtures:

```
[  PASS   ] syn_pass
[  FAIL   ] syn_fail  (intentional failure)
[  FAIL   ] syn_crash_segv  (crash)
[  FAIL   ] syn_hang  (timeout)
[  FAIL   ] syn_known_failure_fail  (known failure behavior)
[   NA    ] syn_na  (not applicable on synthetic vendor)
report: mode=gate | pass=1 fail=11 skip=0 na=1 | flagged=0 | -> report.json
ACCEPTANCE: FAIL — 11 test(s) failed the gate
```

The authoritative "harness behaves correctly" check asserts the full matrix:

```bash
./build/unit_tests                                    # unit checks (reduce/classify, policy CSV, ssh/srun builders)
bash scripts/selftest.sh ./build/hva .  # end-to-end checks against the synthetic oracle
```

The 2-node synthetic tests run on one host with `-n 2 --filter 'syn2_*'`.

---

## Reference

### Policy file (`policy/<gpu>_<nic>.csv`)

Per-platform "simplified CSV" (auto-selected as `<gpu>_<nic>.csv`, e.g.
`mi350_thor.csv`, with fallback to a bare `<nic>.csv`). Columns: `test,class,when,expected,ticket,reason`
(split on the first 5 commas; `reason` is a free-form remainder that may contain
commas). `#` lines are comments; `test` may be an fnmatch glob. Full guide:
[docs/POLICY_REFERENCE.md](docs/POLICY_REFERENCE.md).

- `class` in `MUST_PASS | NA | SKIP | KNOWN_FAILURE`. A registered test with no
  matching row is **not run at all** (silently excluded — the policy is the
  authoritative run list); use `--lint` to find tests missing from the policy.
- `when` (optional) gates the row on the environment: `rocm>=7.1`, `driver<6.0`,
  `gpu<2`, `sym:hipStreamGetId`, `!sym:libionic` (`;`-join = AND).
- `expected` (`KNOWN_FAILURE` only): the expected reduced status (default `fail`).
- A missing policy for the detected platform (`<gpu>_<nic>.csv`, then `<nic>.csv`)
  => **discovery mode** (all tests `MUST_PASS`, gate active) — the seed for a new
  platform file.

### Report (`report.json`)

The machine artifact (`schema_version: "1.3"`). Top level: `harness`
(`{version, git_sha}` — the build-time harness version and git SHA, also
printed by `hva --version`; `git_sha` is `"unknown"` when built with no `.git`,
e.g. from a source tarball), `mode`, `launch` (`{np, ppn, hosts}` — the run's
`-n` / `-ppn` / `-hosts` shape), `summary` counts, per-node `environment`
and `topology` (`{hca, gpu, hca_source, gpu_source, local, note}` — the NIC
and HIP GPU the rank used, each source `user` / `affinity` / `default`, `local`
when the NIC is the GPU's nearest, and `note` why a fallback was taken),
and `tests[]` with `id, category, class, status, reason, ticket, reduced_from,
perf, gate_fail, flagged, note, duration_ms`. Under `--kind perf`, each test's
`perf` map carries `<metric>.{median,min,max,avg,unit,iters,samples}`.
`environment.rocm_version` is resolved via a priority chain (loaded HIP
runtime's own tree, then `$ROCM_PATH`, then `hipRuntimeGetVersion()`, then
`/opt/rocm`); the winning source is recorded in
`environment.rocm_version_source` and its root in `environment.rocm_path`.

### Further docs

- [docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md) — add a HIP/Verbs test + template.
- [docs/NEW_PLATFORM.md](docs/NEW_PLATFORM.md) — first run on new hardware.
- [docs/POLICY_REFERENCE.md](docs/POLICY_REFERENCE.md) — policy CSV format.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design & key decisions.

### Agentic skills

Repo-local Cursor skills under [.cursor/skills/](.cursor/skills/) wrap the whole
lifecycle — from "what's missing" through implement, run, and explain:

| Skill | Role |
|---|---|
| `extract-hip-requirements` | scan source vs a knowledge base + existing tests → punch-list of missing HIP tests (`proposed_tests.json`) |
| `extract-verbs-requirements` | same for Verbs (`ibv_*`/`rdma_*`/`mlx5dv_*`) |
| `acceptance-add-test` | scaffold a test (from a description **or** a `PROPOSE` item) + policy row |
| `acceptance-run` | build + run the requested tests, report the verdict |
| `acceptance-summarize` | summarize a `report.json` (counts, per-category, failures) |

**How to use them.** You do not invoke skills by name — just describe the task in
Cursor and the matching skill triggers automatically (the wording below is only an
example; any phrasing with the same intent works):

| You want to… | Ask Cursor something like… | Skill that fires |
|---|---|---|
| Find missing tests | "what HIP acceptance tests are still missing here?" | `extract-hip-requirements` |
| Run the suite | "run the HIP and Verbs acceptance tests with perf metrics and give me the verdict" | `acceptance-run` |
| Add a test | "add a HIP test that checks `hipDeviceSynchronize` on an idle device succeeds, with a latency metric" | `acceptance-add-test` |
| Explain results | "summarize the `report.json` from that run and show why the gate failed" | `acceptance-summarize` |

**Full loop (minimal input).** The five skills chain end-to-end — one request such
as *"run the tests, find a missing HIP scenario, implement it, run it, and show me
what changed"* drives the whole pipeline:

```
acceptance-run  →  extract-hip/verbs-requirements  →  acceptance-add-test  →  acceptance-run  →  acceptance-summarize
   baseline            proposed_tests.json               scaffold + policy        re-run             what changed
```

`extract-*` writes a `PROPOSE` list; `acceptance-add-test` consumes one item
(`title`→id, `sequence`/`verify`→body, `class`+`gate`→policy row) and, once
committed, the scenario flips from PROPOSE to COVERED on the next `extract-*` run —
the repo itself is the progress tracker. Each skill just drives the same
`./build/hva` CLI (and read-only `python3`/`bash` scans), so anything
a skill does you can also run by hand.

### Performance

`--kind perf` records latency/bandwidth per test into `report.json` (no separate
perf tests; perf is folded into the functional tests). Perf is **report-only**:
it never changes the acceptance verdict or the exit code — the gate stays
"functionally correct or not".

A perf run ends with a Performance block on the console (suppressed by
`--quiet`), grouped category → test → metric, sizes ordered numerically. The
category headers match the functional block's `--- category ---` style, and
each test's metrics get a blank line to themselves:

```
--- performance ---
    metric                        size      median        min        max unit     n

--- hip/datamovement ---
  Memcpy_Size_Sweep
    h2d.sync                    64KiB        2.45       2.31       2.58 GB/s   10
    h2d.sync                   256MiB       54.03      53.88      54.11 GB/s   10

--- verbs/hip_mixed ---
  vat_hipalloc_x_ibvreg__finegrain__regmr
    transfer.lat                          87.482     87.482     87.482 us/op    1
    rank1.alloc_matrix.write_imm           16.74      16.74      16.74 us/op    1
```

A single-pass metric (`n=1`) has no spread, so min/max just equal the median.

On a two-node run every rank's metrics are kept: the lowest rank that measured
anything keeps its keys as-is, the other rank's are prefixed `rank<N>.`.

`--perf-csv <file>` writes the same numbers as
`category,test,metric,size,size_bytes,median,min,max,avg,unit,iters` for
tracking over time. A failed CSV write warns; it does not fail the run.

#### Comparing runs

`scripts/perf_report.py` renders a report in full — every metric with its band,
iteration count and any anomalies — and diffs it against earlier ones:

```bash
./build/hva --kind perf --report new.json --perf-csv new.csv
python3 scripts/perf_report.py new.json                       # render
python3 scripts/perf_report.py new.json --baseline old.json   # diff
```

A metric is called `worse`/`better` only if its median moved more than
`--threshold` (default 10%, tune it with `--threshold PCT`) **and** the two
runs' `[min, max]` bands do not overlap; everything else is `~` (within noise).
Unknown units get a delta but no verdict.

By default the comparison table only lists `worse`/`better` rows — pass
`--show-all` to see every shared metric, including the `~` and `?` ones.

Pass `--baseline` more than once to fold several runs into one baseline — the
medians are combined and the band becomes their envelope. Do this whenever you
care about the answer: a single baseline run only knows how much a metric
wobbled *within* that run, which on a shared machine is much less than it
wobbles *between* runs, so a one-run baseline reports ordinary drift as a
regression. Measured on an MI300X box, `d2d.sync` at 16MiB moved 1071 → 603 GB/s
across two runs of the *same* binary, each run internally tight — three folded
baseline runs cover that; one does not.

```bash
python3 scripts/perf_report.py new.json \
        --baseline r1.json --baseline r2.json --baseline r3.json \
        --threshold 15 --fail-on-regression
```

`--fail-on-regression` makes *the script* exit 1 — a CI knob on the diff, not on
the suite's gate. `--format md` emits Markdown for a ticket or PR.

Raw metrics straight out of the report, if you want them unrendered:

```bash
python3 -c 'import json; d=json.load(open("report.json"));
[print(t["id"], t["perf"]) for t in d["tests"] if t.get("perf")]'
```
