# HIP/Verbs Acceptance Suite — Architecture

**Status:** harness, the ssh multi-node launcher, the HIP and Verbs functional
test suites, and per-test performance measurement are all **implemented and
hardware-validated**.
**Scope:** deployment + test infrastructure. Functional correctness is the
primary signal; performance metrics are collected opt-in via `--kind perf|all`
(§9).

> **Status (implemented).** The framework-agnostic harness (registry, EXEC
> isolation, watchdog, policy/classification, Reporter, SocketBootstrap) plus the
> HIP and Verbs test suites are built and hardware-validated: a synthetic
> self-test exercises every failure mode (pass/fail/skip/crash/timeout/corrupt/
> known-failure/NA) and the 2-rank path was validated **across two real nodes**.
> Notable design settlements: (a) the EXEC child result channel is the harness's
> own **key=value** `TestResult` file (not JSON — only the final `report.json` is
> JSON); (b) `SocketBootstrap` listener binds `0.0.0.0` and the connector resolves
> hostnames via `getaddrinfo` (§4 — found necessary on real 2-node hardware);
> (c) the ssh **`--nodes` multi-node launcher** returns each worker's results on
> its stdout; the launcher must run on a host with a runtime compatible with the
> binary (see §13).

> **Key design decisions** (all implemented and node-validated):
> - test framework is **custom-minimal, not GoogleTest** (§6);
> - policy is **per-vendor simplified-CSV** (`policy/<vendor>.csv`), the report is
>   **hand-emitted JSON**, no third-party JSON library (§7.1, §8, §10);
> - the report **always carries full per-test detail** (no `--verbose` knob) (§8);
> - **4-class taxonomy** — `MUST_PASS` / `KNOWN_FAILURE` / `SKIP` / `NA`; no
>   `CONDITIONAL` (applicability is policy data), no `UNCLASSIFIED` (unlisted = not
>   run), no `REGRESSION` (folded into `MUST_PASS`);
> - policy rows carry an optional **`when` predicate** for version/capability
>   applicability (§7.1a);
> - **every test is isolated by default**; `KNOWN_FAILURE` outputs as `Fail` and
>   gate-fails; the policy file is the authoritative run list.

> **Design notes.** (1) `SocketBootstrap` is scoped to 1/2 ranks — it is net-new,
> not a "generalization" of the 2-peer probe (§4); (2) cross-rank verdict
> reconciliation is an explicit Reporter-owned reduction, not an `MPI_Allreduce`
> (§7.3); (3) an absent policy does not fail new hardware silently — it triggers
> **discovery mode** (WARN + all tests treated as `MUST_PASS`, gate active), and
> tests not listed in a present policy are simply **not run** (§7.2);
> (4) the process-isolation / EXEC description matches the code and keeps a single
> result channel (§6.3). Open dependencies are called out in §13.

---

## 1. Purpose

A standalone acceptance suite that exercises the **HIP** and **Verbs** primitives
RCCL depends on, **at the libibverbs / HIP runtime level — with no RCCL in the
build**. It runs *before* the main test suite to catch fundamental HIP/Verbs
failures early, and serves as the acceptance gate when validating **new hardware
or firmware**: passing the suite is the criterion that confirms basic HIP/Verbs
scenarios are supported on a new cluster.

The suite also doubles as:

- A **requirements baseline** for HIP and ionic/Thor2 functionality (e.g. a
  feature enabled for Mellanox at the NCCL level but missing on ionic surfaces
  here as an `NA` entry — flipped to `MUST_PASS` when the vendor implements it).
- A **known-issues / regression registry** — every `NA` or `KNOWN_FAILURE`
  behavior is recorded with a reason and a ticket.

Functional correctness is the priority. Performance measurement is folded into
the same tests and collected opt-in under `--kind perf|all` (§9), so it never
slows the functional gate.

---

## 2. Design constraints (from the spec, verified against the codebase)

| Constraint | Consequence in this design |
|---|---|
| **No RCCL dependency** — binary must build/run on a bare cluster node | New, RCCL-free `common/` tree; standalone CMake project (§5, §6) |
| **MPI optional**; ssh/sockets is a primary path | `Bootstrap` abstraction; `SocketBootstrap` is the only backend today, `MpiBootstrap` is a designed-but-unimplemented seam (§4) |
| Reports **passed / failed / skipped(reason)** counts | own `Status{pass/fail/skip/none}` per test + hand-emitted JSON, reconciled/classified by the Reporter (§7, §8) |
| **Detect missing symbols**; version handling for HIP / verbs / driver | `Capability` probe layer using `dlopen`/`dlsym`; versions recorded as report dimensions (§5) |
| A hang/timeout/crash is the **regression signal** | Per-test process isolation + watchdog (reuse the existing `ProcessIsolatedTestRunner` pattern) (§4, §6) |
| Hardware-specific (ionic, Thor2, Mellanox) + a generic set | Vendor auto-detected at runtime; per-vendor classification matrix (§7) |
| Run on single and multi node (loopback and non-loopback) | Same test code over the `Bootstrap` abstraction; topology is test metadata (§4, §6) |

### What we reuse vs. what is net-new

The design deliberately follows existing patterns, but review established that
several pieces are **borrowed ideas, not drop-in reuse**. This is stated plainly
so the team budgets the net-new work correctly.

- **Pattern to follow (structure reuse):**
  `test/common/{MPIEnvironment,MPITestBase,MPITestCore}.hpp` — a GoogleTest
  binary with a global `::testing::Environment` for bootstrap/teardown and
  layered `TestBase` classes. We reuse this *structure*, but strip the RCCL
  coupling (`MPITestCore` includes `rccl/rccl.h` and creates an `ncclComm`) and
  abstract the transport.
  **Net-new:** the existing cross-rank aggregation is `MPI_Allreduce` of a single
  `int retCode` (`MPIEnvironment.cpp:282-290`) — one boolean for the whole
  process. It **cannot** carry the per-test, skip/NA-aware verdicts this suite
  needs, so verdict reconciliation is redesigned in the Reporter (§7.3), **not**
  inherited from that allreduce.
- **Crash/hang containment (pattern reuse):**
  `test/common/ProcessIsolatedTestRunner.cpp` provides a re-exec + watchdog
  (SIGTERM→SIGKILL) pattern we follow for isolation. **Correction from review:**
  it is driven *from inside* a gtest `TEST()` body, selects the child via the env
  var `RCCL_PIT_REEXEC_TEST` (not a CLI arg), derives the child filter from the
  enclosing test, and stores results in a private struct — it does **not** emit
  gtest JSON for children. We therefore do not reuse it verbatim; see §6.3 for
  the integration we actually adopt.
- **No-MPI OOB rendezvous (idea reuse):** the RCCL repo's
  `test/transport/qp_link_probe.c` exchanges QP metadata over TCP (port 18515).
  **Correction from review:** it is a hardcoded **2-peer** rendezvous (one
  `accept`, fixed port, `connect` retries forever with no timeout, barrier valid
  only for two processes) — it has no rank/size/allgather. `SocketBootstrap`
  borrows the TCP-OOB idea but is **net-new code**, and is scoped to exactly what
  the test set needs (1 or 2 ranks; see §4).
- **Test content is implementation-grade already:** the RCCL repo's
  `test/transport/verbs_acceptance/VAT_TEST_SPEC.md` specifies 27 verbs tests
  with exact net_ib_cast-derived parameters and a feature→test map (24 are
  single-node/loopback; only 8.1/8.2/8.3 are 2-node, all strictly pairwise). The
  identified HIP/verbs scenarios become the test bodies; this document covers the
  harness they plug into.

---

## 3. High-level shape

A **single self-contained C++ binary** (no GoogleTest, no third-party JSON).
It runs in one of three modes, selected by argv:

```
            ┌──────────────────────────────────────────────────────────┐
            │  hva  (one binary, three modes)                            │
            ├──────────────────────────────────────────────────────────┤
  --nodes   │  LAUNCH mode  → parse CLI, ssh-deploy, start workers,      │
            │                 collect per-node JSON, merge → report      │
            ├──────────────────────────────────────────────────────────┤
  --worker  │  WORKER mode  → SocketBootstrap (rank/size/rendezvous),   │
            │                 run selected tests, emit node-local JSON   │
            ├──────────────────────────────────────────────────────────┤
  --exec ID │  EXEC mode    → run ONE test, process-isolated, watchdog   │
            │                 (used by WORKER to isolate each test)      │
            └──────────────────────────────────────────────────────────┘
```

The **deliverable/product is flat**: the single `hva` binary plus the
per-vendor `policy/*.csv` files. Nothing else is packaged or placed on a node.
Everything else (device, vendor, gid, port, iface/BDF, ROCm/HIP/driver/FW
versions, GPU count) is auto-detected at runtime.

---

## 4. Bootstrap abstraction (the key decoupling)

Two concerns are kept separate:

1. **Process launch / placement** — pluggable *launcher*: `ssh` or `srun` now
   (`--remote-launcher ssh|srun`), `mpirun` later.
2. **Runtime out-of-band (OOB) exchange** — pluggable *bootstrap*: how ranks find
   each other and swap QP metadata at runtime.

Tests never touch MPI or sockets directly; they talk to the `Bootstrap`
interface. As a result, single-node loopback, two-node socket, and (future)
mpirun runs all use the **same test code**.

**Scope decision (from review): the default path supports `size ∈ {1, 2}` only.**
The VAT set is 24 single-node/loopback tests plus 3 strictly *pairwise* 2-node
tests — there is no test that needs an N-rank collective. Building a general
star-topology allgather to serve three pairwise tests would be the hardest and
least-justified component in the suite. So `SocketBootstrap` implements:

- `size == 1` — loopback; `barrier`/`exchange` are no-ops, `allgather` is a copy.
- `size == 2` — one rank-0 listener, one rank-1 connector (with a **bounded**
  connect/accept timeout, unlike the probe's infinite retry); `exchange` is the
  QP-metadata swap; `allgather` is a 2-way swap; `barrier` is a counted
  round-trip. **One persistent connection is reused for all barriers/exchanges**
  (nit) — unlike `qp_link_probe.c`, which opens a fresh port per sync
  (`TCP_PORT+1+counter`). A single long-lived socket means exactly **one** TCP
  port to allow through firewalls and no per-barrier port-range growth.

`allgather` is kept in the interface (it is what the global Reporter rendezvous
uses) but is only required to be correct for `size ≤ 2` in this iteration. True
N-rank support is deferred to the same bucket as the mpirun launcher (§13); the
`ssh`/`srun` launchers only place one process per node each — they do not, by
themselves, make `SocketBootstrap` N-rank. `MpiBootstrap`, when present, is
naturally N-rank and can be used for larger runs.

```cpp
// common/Bootstrap.hpp
class Bootstrap {
public:
  virtual int  rank() const = 0;
  virtual int  size() const = 0;
  virtual void barrier() = 0;
  // collective: gather `len` bytes from every rank into `out` (size*len).
  // SocketBootstrap: required correct for size<=2 this iteration.
  virtual void allgather(const void* in, void* out, size_t len) = 0;
  // pairwise: swap `len` bytes with `peer` (the QP-metadata exchange primitive).
  virtual void exchange(int peer, const void* send, void* recv, size_t len) = 0;
  virtual ~Bootstrap() = default;
};
```

For a 2-node test, the pairing is unambiguous (rank 0 ↔ rank 1). The suite does
**not** support choosing which two of N ranks pair up — that need does not exist
in the current test set and is part of the deferred N-rank work.

**Non-goal: cross-rank pass/fail consensus.** `barrier`/`exchange`/`allgather`
are **data-plane OOB primitives** (rendezvous + QP-metadata swap) only. The suite
deliberately does **not** layer a cross-rank pass/fail *consensus* on top of them
(i.e. no `ASSERT_BOOTSTRAP_*` macro family, no socket allreduce-of-status).
Coordinated exit when a peer fails is handled by the per-test watchdog (§6.3), and
the merged verdict by the Reporter (§7.3) — not by per-assertion collectives.

> *Why not the MPI model.* The existing `ASSERT_MPI_*` consensus
> (`TestChecks.hpp:408-574`) — each assertion does `MPI_Allreduce(MIN)`, the
> failing rank `FAIL`s and the rest `GTEST_SKIP` — is elegant in MPI because an
> N-rank collective substrate already exists and the suite accepts whole-job
> `MPI_Abort` as its backstop (`MPIEnvironment.cpp:269`). Neither holds here:
> we are scoped to size ≤ 2, we *require* a per-test merged JSON (a whole-job
> abort would destroy it), and our primary signal is crash/hang/timeout — which a
> collective **cannot** observe (a dead rank never reaches the allreduce, so the
> peer blocks forever; that is exactly why MPI needs its timeout+abort underneath
> the elegant layer). Porting consensus would add a second collective stack to
> grace-handle the *cooperative-failure* path of just 3 of 27 tests, while the
> watchdog must exist regardless and already covers that path. Intentionally not
> ported; see §6.3 and §7.3.

| Implementation | When | Notes |
|---|---|---|
| `SocketBootstrap` | **default** | Net-new TCP rendezvous (borrows the OOB idea from `qp_link_probe.c`). Scoped to size ≤ 2 with bounded timeouts. Rank 0 is the listener; rank/size/rendezvous passed to workers as CLI args by the launcher. |
| `MpiBootstrap` | optional | Compiled in only when `find_package(MPI)` succeeds (`#ifdef ACCEPT_HAVE_MPI`). N-rank capable; reuses `mpirun` placement where MPI exists. **Not yet implemented.** |

> **Implementation note (validated on real 2-node hardware).** Two things were
> required for cross-host rendezvous that single-host testing (numeric
> `127.0.0.1`) masked: (1) the **listener binds `0.0.0.0`**, *not* the rendezvous
> host — a node resolving its own FQDN often hits `127.0.1.1` (Ubuntu
> `/etc/hosts`), which would bind loopback and be unreachable from the peer; the
> rendezvous host is meaningful only to the connector. (2) The **connector
> resolves hostnames via `getaddrinfo`** (IPv4), not just `inet_pton` — otherwise
> only numeric IPs work. Both are in `common/SockCompat.cpp` / `SocketBootstrap.cpp`.

**Launch flow (ssh / socket — the default path).** *Built (Phase 1b) and
node-validated. Rank→worker args are passed on the **CLI**
(`--rank/--size/--rendezvous --results-out -`), not env; per-rank results come back
on each ssh child's stdout. Step 1 below is currently same-path/NFS (scp-deploy for
non-shared-FS is deferred — §13).*

1. LAUNCH mode resolves the node list (`--nodes`) and runs each node's binary at
   `--remote-binary` (default: the launcher's own path — same-path/shared-FS
   assumption; scp-deploy is deferred, §13).
2. Starts each node's binary in WORKER mode, passing `rank`, `size`, and the
   rank-0 `rendezvous host:port`.
3. Workers construct `SocketBootstrap`, rendezvous, run their tests, and stream
   their node-local result set back on stdout (`--results-out -`).
4. LAUNCH mode collects the per-node results and the Reporter merges them.

---

## 5. Environment & capability layer ("dimensions")

The `Capability` module probes each node at startup and produces an **environment
descriptor** that is (a) recorded into the report as dimensions and (b) the input
to gating decisions.

Detected automatically — never configured:

- **NIC vendor** — from device-name prefix, mapped to a policy key: `mlx5_*` →
  `connectx`, `bnxt_re*` → `thor`, `ionic_*`/`rdma_*` → `ionic`
  (`Capability.cpp`). The key selects the policy file (§7).
- **Device / port** — first active RoCE device + port.
- **gid_index** — selected from the GID table, *not* hardcoded. **Best-effort,
  not authoritative (per review):** the GID-table layout and the RoCEv1/v2
  distinction differ across mlx5 (exposed via
  `/sys/class/infiniband/<dev>/ports/<p>/gid_attrs/types/<i>`), bnxt_re, and
  ionic — e.g. the RoCEv2 GID can land at a different index on each driver, so
  the *rule* is not uniform.
  When multiple RoCEv2 GIDs exist (multiple IPs/VLANs), the selector prefers the
  GID whose netdev/IP matches the address the launcher used to reach the peer.
  A **`--gid <index>`** override (implemented; also `ACCEPT_IB_GID_INDEX`) is the
  escape hatch when auto-selection picks wrong.
- **iface / PCIe BDF** — from sysfs (`/sys/class/infiniband/<dev>/device` → BDF;
  associated netdev → iface). Needed by link-down/nicctl tests; not configured.
- **Versions** — ROCm/HIP runtime + driver, NIC driver/FW. Recorded dimensions.
- **GPU count** — recorded as a dimension. **Rank→GPU binding is not implemented**
  in the harness: `EnvironmentInfo` carries no `local_rank`/`gpu_bound`, and tests
  do not auto-bind a GPU by local rank. With size ≤ 2 and one rank per node (the
  common case) tests use GPU 0; explicit multi-GPU binding is future work (§13).
- **Optional symbols, present/absent** — via `dlopen`/`dlsym`:
  `libionic` (ionic direct-verbs), `hsa_amd_portable_export_dmabuf`, optional
  HIP/cuMem entry points. This implements the spec's `Symbol_Version_Detection`
  and `DriverEntryPoint_Resolve` requirements directly, and feeds gating.

**Linking strategy.** The binary links **no** RDMA/HIP libraries. `libibverbs`,
`libamdhip64` and optional symbols are all resolved at runtime via `dlopen`/`dlsym`
through thin shims in `wrapper/`, so the binary loads and reports cleanly on a node
missing them rather than failing to start. (`-ldl`, `pthread` are the hard deps.)

**Device selection is deterministic and logged:** "first active RoCE device" is
defined as the lowest-named active RoCEv2 port, and the chosen device is recorded
per node. Nodes with multiple RDMA devices use a **`--device <name>`** override
(implemented; also `ACCEPT_IB_DEVICE`) to disambiguate.

The cluster is assumed **homogeneous**, but "homogeneous" is treated as a
*rollup, not a precondition* (per review): each node's descriptor is recorded
individually in the report (§8 `environment` is **per node** with a
`homogeneous: bool` rollup), and a mismatch is **flagged, not necessarily
fatal** — a heterogeneous cluster still produces a usable, per-node report.

---

## 6. Test framework internals

> **Framework: custom-minimal, not GoogleTest (decided, not deferred).**
> The harness has **no GoogleTest**: tests are plain functions
> `TestResult(TestContext&)` registered via an `ACCEPT_TEST(...)` macro into a
> `TestRegistry` (not `TEST_F`/`ACCEPT_TEST_META`); there is a
> `TestContext{rank,size,Bootstrap*}` instead of a gtest `Environment`. This was a
> deliberate decision: the regression signal is
> **out-of-process** (fork+watchdog), which every in-process framework fights;
> gtest's real benefits (familiarity, assertion diagnostics, `TEST_P`) are small
> and ~50 LOC to replicate. The names below (`AcceptEnvironment`, `ACCEPT_TEST_META`,
> `TEST_F`) are the *original gtest-target* spelling, kept for traceability; the
> *roles* (registry/metadata, isolation, single result channel) are implemented in
> the custom form. **Low lock-in:** a gtest adapter could be added later (a gtest
> test = one more registered test kind) if a compelling need emerges.

### Test bodies & shared helpers (as-built)

There are **no `TestBase`/fixture class hierarchies and no gtest
`::testing::Environment`.** A test is a plain function
`TestResult(TestContext&)` registered via `ACCEPT_TEST(...)`. Shared setup lives
in free helpers, not base classes:

- `verbs/` — device/PD/CQ/QP/MR + TCP rendezvous helpers (the `vat_common`
  surface), including the mixed HIP+verbs tests (`verbs/hip_mixed.cc`,
  `verbs/hip_coherence.cc`, …).
- `hip/utils/` — `hipMalloc`/stream/event/kernel helpers and the perf memcpy
  helpers.
- `wrapper/` — the `dlopen`/`dlsym` shims for `libibverbs`/`libamdhip64`.

The worker constructs the chosen `Bootstrap` and runs the capability probe once;
it does **not** aggregate verdicts across ranks — each rank emits its own complete
result set and cross-rank reconciliation is the Reporter's job (§7.3).

### Intrinsic test metadata (in code, fixed — never varies by cluster)

Registered with each test via a small macro:

```cpp
// TestRegistry.hpp — as-built signature
ACCEPT_TEST(vat_device_enum,          // unique id
            "verbs/foundation",       // category (subsystem taxonomy)
            Topology::Single,         // Single | Loopback | TwoNode
            Kind::Functional,         // Functional | Performance
            /*timeout_sec=*/30) { ... return pass(); }
```

These are properties of the *test itself* (topology, kind, how long it may run).

> **Not yet implemented.** `required_caps`/`feature_tags` (and the `CAP_*`
> capability gating they would feed) are **not** part of the built `TestMeta`, and
> there is **no `--dump-requirements`** mode. `Kind::Performance` also exists but
> no test registers with it — perf is toggled globally by `--kind perf|all` (§9),
> not by per-test kind.

### 6.3 Process isolation, EXEC mode & a single result channel

**Corrected from review.** `ProcessIsolatedTestRunner` is not reused verbatim: it
runs *inside* a gtest test body, selects children via the env var
`RCCL_PIT_REEXEC_TEST`, and collects results into a private struct that is
**never written as gtest JSON**. Adopting it as-is would create two result
channels (in-process tests in the gtest JSON; isolated tests — including the
crash/timeout cases that matter most — outside it), which breaks the §8 merge.

We adopt the cleaner integration instead:

- The **WORKER does the forking itself**. Every test is isolated by default
  (`--isolate all`, the default; `--isolate none` runs in-process for
  single-test debugging — the launcher also picks `none` automatically when exactly
  one test is selected and `--isolate` was not given). **Carve-out:** `TwoNode`
  tests are **never** isolated (a forked child cannot share the live rendezvous
  socket) and always run in-process; conversely, a few tests that must be isolated
  for correctness (e.g. `hip/ipc`, device-reset, `vat_fork_init`) stay isolated
  even under `--isolate none`. For each isolated test the worker `fork`/`exec`s the
  same binary in **EXEC mode** — `hva --exec <test_id>` (a real CLI
  arg, not an env marker). **EXEC runs exactly one test per process**, so a child's
  result file contains exactly one row. The watchdog escalates
  SIGTERM→SIGKILL (the timing/escalation pattern is borrowed from
  `ProcessIsolatedTestRunner.cpp:532-558`).
- **The child's exit status is authoritative; its result file is corroborating
  (corrected per review).** The EXEC child writes its `TestResult` as a small
  **key=value file** (`--result-file` — *not* JSON; only the final
  `report.json` is JSON) and only on normal exit, so a crashed/SIGKILLed child may
  leave **no file *or* a truncated/partial one** — the worker must not trust file
  *presence*. The rule:
  - child exit code maps to outcome — `0`→ trust JSON; non-zero/`-signal`→ crash,
    timeout code→ timeout;
  - the child JSON is used **only if it parses *and* contains the expected
    `test_id`**; otherwise (missing, truncated, or wrong row) the worker
    **synthesizes a `fail` row** (status `fail`, reason `crash`/`timeout`) from the
    exit status. That synthesized row *is* the crash/hang gate signal.
  This yields exactly **one result schema** for every test regardless of isolation,
  and is robust to partial output, not just absent output.
- Non-isolated tests run in-process in the same WORKER process and emit their rows
  directly. Isolation is therefore an orthogonal execution detail, not a second
  framework.

A fresh device/context open per isolated test (a property the fork model gives us
for free) also bounds RDMA/GPU resource leakage from expected crashes; see §13
for the between-test health-probe follow-up.

**The watchdog is also the coordination mechanism for the 3 pairwise tests**
(8.1/8.2/8.3) — not just for crashes. If one peer fails an assertion and exits,
the other peer, blocked in a subsequent verbs operation (a CQE/recv that will
never arrive), is bounded by its per-test `timeout_sec` and reaped via
SIGTERM→SIGKILL; the Reporter then synthesizes its row from exit/timeout status.
This **intentionally replaces a consensus-at-assertion scheme**: it costs at most
one extra test-timeout on the *rare cooperative-failure path of 3 of 27 tests*, in
exchange for not building and maintaining a second collective+macro stack (§4
non-goal). The verdict is identical either way (the test fails); only the healthy
peer's exit is slower and via signal rather than a clean `GTEST_SKIP`.

---

## 7. Classification & gating

Classification is **a function of (vendor, test)** — it varies by hardware. It is
kept **out of the code** and **out of the deployment CLI**, in declarative data
files that double as the requirements/known-issues deliverable for the HW teams.

Classes (4): `MUST_PASS` / `KNOWN_FAILURE` / `SKIP` / `NA`.
There is **no `UNCLASSIFIED`** — a test with no policy entry is simply **not run**
and does not appear in the report; the policy file is the authoritative
run list. There is **no `CONDITIONAL`** — applicability is expressed as data
(`NA`/`SKIP` + reason, flipped to `MUST_PASS` when firmware adds the feature), so
test bodies carry no capability-probe/skip branches. There is **no
`REGRESSION`** — it had gate semantics identical to `MUST_PASS` and existed only to
trigger subprocess isolation; isolation is now the default for every test,
so crash/hang/timeout tests are simply `MUST_PASS`. The legacy `REGRESSION` token is
still accepted on read and mapped to `MUST_PASS`.

### 7.1 `policy/<vendor>.csv` — per-vendor simplified CSV

One CSV file **per vendor** (the filename is the vendor; the vendor column is
dropped). Columns: `test,class,when,expected,ticket,reason`. **Parse
rule:** split on the **first 5 commas only** — `reason` is a free-form last field
that may itself contain commas, so **no quoting/escaping** is needed (this is
"simplified CSV", not RFC-4180; non-last fields are identifiers/enums/tickets and
never contain commas). `#` lines are comments; `fnmatch` glob on the `test` column.

**Column values:**
- `class` — `MUST_PASS | KNOWN_FAILURE | SKIP | NA` (case-insensitive on read;
  a test **not listed ⇒ not run**; the legacy token `REGRESSION` is accepted and
  mapped to `MUST_PASS`).
- `when` — optional version/capability predicate. Empty = always applies.
  See §7.1a for the predicate grammar. If a test is listed but a row's `when`
  predicate fails on a **software** dimension (`rocm`/`driver`/`fw`/`sym:`) the
  engine returns **`SKIP`** (upgrade signal); a failing **hardware** dimension or an
  explicit `NA` row returns **`NA`**.
- `expected` — **only meaningful for `KNOWN_FAILURE`** (ignored for other classes).
  Values: `fail | pass | skip | none`. **Empty ⇒ `fail`**.
- `ticket`, `reason` — free-form (reason is the comma-tolerant last field).

```
# policy/ionic.csv  — columns: test,class,when,expected,ticket,reason
vat_relaxed_ordering,NA,,,TICKET-123,RO unsupported on ionic fw (flip to MUST_PASS when fw adds it)
vat_rc_qp_reset_recovery,KNOWN_FAILURE,,fail,TICKET-450,in-place RESET broken on ionic
vat_atomic_fetch_add,NA,,,,atomics not supported on this NIC/driver
vat_gdma_fabric,MUST_PASS,rocm>=7.12,,TICKET-77,fabric API requires ROCm 7.12
```

```
# policy/connectx.csv   (mlx5_* NICs)
vat_relaxed_ordering,MUST_PASS,,,,
vat_ionic_*,NA,,,,               # wildcard: ionic-only tests are N/A on ConnectX
```

- **Why per-vendor:** clean per-file ownership (each NIC's policy lives in its
  own `<vendor>.csv`), simple diffs/blame with no shared-file merge contention, and the new-hardware
  flow falls out for free — **a missing `policy/<vendor>.csv` for the detected
  vendor = discovery mode** (§7.2). The unified cross-vendor matrix (the deliverable
  *view*) is rendered on demand (`--dump-requirements`), not stored monolithically.
- **No `defaults`/`inherit_defaults`:** the policy file is the run list — an
  unlisted test is **not run** (removes the earlier opt-out footgun and a whole
  precedence tier). Within a file: exact id → wildcard (longest-literal-prefix wins).
- **`NA`** = not applicable for this vendor: not run, recorded *not-applicable*,
  distinct from skip. The Reporter **synthesizes** `NA` rows from the policy (they
  never run). `NA`/`SKIP`/`KNOWN_FAILURE` are Reporter-level classifications, not
  rank-level statuses (§8).
- **`KNOWN_FAILURE` + `expected`** — how spec step 11 ("reproduce known real
  failures") is tracked. `expected` (default `fail`) is recorded and the detail
  distinguishes "known failure" (reduced status matched) from "known failure
  possibly fixed" (it did not — update the policy). **Gate: `KNOWN_FAILURE` always
  outputs `Fail` and gate-fails** (same pressure to fix as `MUST_PASS`;
  the only exception is a reduced `none`, which becomes `NA`). This reverses the
  earlier "never gate-fails" design.
- Each row carries **`ticket` + `reason`** → the files *are* the living known-issues
  registry.
- **Reserved (not v1):** a `fw_version` column — firmware updates change behavior,
  so an `NA`/`KNOWN_FAILURE` becomes `MUST_PASS` once a fix ships.

### 7.2 Absent policy → discovery mode

The suite's #1 use case is onboarding **new hardware/firmware** — precisely the
case where no policy exists yet. Behaviour by file presence:

- **No `policy/<vendor>.csv` for the detected vendor** (and `--policy` not passed)
  → **discovery mode**: a loud `WARN` banner, `mode: "discovery"` in the report,
  and **every registered test runs with `MUST_PASS` semantics — the gate stays
  active**. This is stricter than the old "gate-off" discovery: new hardware
  surfaces failures immediately instead of silently collecting outcomes. (A file
  passed **explicitly** by path but not found is a **hard error** — typo guard.)
- **File present but a test is unlisted** → the test is **not selected** at all:
  it does not run and does not appear in the report. `--lint` cross-references the
  registry against the policy and lists such gaps, exiting non-zero if any exist.

There is no `defaults`/`inherit_defaults` tier and no `UNCLASSIFIED` outcome —
file-exists vs not, and listed vs not, fully determine behaviour.

### 7.3 Cross-rank verdict reconciliation (Reporter-owned)

This is the suite's core output and is **not** inherited from the existing
`MPI_Allreduce(MAX)` (which yields one boolean per process). The Reporter is the
sole authority:

1. Every rank writes its **complete** per-test results as the harness's own
   **key=value** result file (`rank=N` header + `@env`/`@test` blocks — *not*
   JSON, so nothing has to parse JSON). Only the merged `report.json` is JSON.
2. A rank-level result is one of **`pass | fail | skip | none`** only. `none` is
   the **non-participating / idle** outcome (gap from review): for a pairwise test
   where, by design, only one rank drives and the peer is a passive target (e.g.
   the passive-receiver pattern in VAT 4.4, or the idle side of 8.2), the idle
   rank reports `none` — **not** `skip`. This matters because of rule 4: a
   genuinely idle rank must not look like a skipped MUST_PASS test. `none` is a
   first-class value in our `Status` enum, so a passive rank just returns it
   (`none_()`); no special marker needed.
3. The Reporter reduces, **per test**, over rank results using a defined lattice
   where `none` is the **identity** (ignored):

   `fail > skip > pass`  ·  `none` = identity (drop before reducing)

   - drop all `none` results; if none remain, the test is `na` (no rank
     participated — Reporter-synthesized, see below);
   - else any remaining rank `fail` → test `fail`;
   - else any remaining rank `skip` → test `skip` (carry the reason);
   - else `pass`.

   Note: `na` is **not** a rank-level result — it is a Reporter classification
   outcome (a test not-applicable for the vendor is never run, §7.1). It only
   appears as the *reduced* status, never in a rank's JSON.
4. **MUST_PASS + any participating rank skipped → CI failure** (the spec's
   "unexpected skip of a MUST_PASS test is a failure", now a *cross-rank* rule
   over participating ranks only).
   > **Maintainer warning (do not re-introduce the MPI coordination-skip).** Each
   > rank must report its **honest** local outcome — `none` if passive (rule 2),
   > `pass`/`fail`/`skip` only for a genuine local result. A rank must **never**
   > emit `skip` merely because a *peer* failed (the `ASSERT_MPI_*` pattern, where
   > the healthy rank `GTEST_SKIP`s on a sibling's failure). Doing so would collide
   > with this rule: the coordination-skip would be misread as an unexpected
   > MUST_PASS skip and double-count the peer's failure. The lattice already
   > derives the correct merged verdict (`fail` wins) from honest per-rank results
   > — no coordination skip is needed or wanted. (This is why §4 rejects an
   > `ASSERT_BOOTSTRAP_*` family.)
5. **Classification is applied *after* reduction** (nit): the lattice produces a
   reduced status (`pass`/`fail`/`skip`/`na`); the class
   (`MUST_PASS`/`KNOWN_FAILURE`/`SKIP`/`NA`) is then applied to that single reduced
   status to compute the output status + gate verdict per §7.4 — so §7.3 and §7.4
   are one model, not two. `KNOWN_FAILURE` always outputs `Fail`/gate-fail
   (except reduced `none` → `NA`); the detail records whether the known failure
   reproduced or "possibly fixed". Unlisted tests are not run and do not appear in
   the report.
6. Ranks should never disagree on *classification* (it is keyed on vendor and the
   cluster is homogeneous); if they do, the Reporter records it as a config error.

### 7.4 Class → gate semantics (4 classes)

Applicability is decided in policy (no in-test capability probe). `skip` is a rare
*runtime* outcome (run-shape only; mostly avoided by selection), and a `skip` from
a `MUST_PASS` test is a gate failure (spec: unexpected skip of a must-pass test).
Every test runs process-isolated by default, so a crash/hang/timeout is
always contained and reported as the test's outcome.

| Class | When | Runs? | Output status | Gate |
|---|---|---|---|---|
| **MUST_PASS** | always applicable | yes (isolated) | `pass` / `fail` / `fail` (unexpected skip) | `fail` gate-fails |
| **SKIP** | software-gated (`when` predicate on version/sym dim fails) | **no** | `skip` + reason (upgrade signal) | neutral |
| **KNOWN_FAILURE** | known bug (tracked ticket) | yes (isolated) | **`fail`** — with detail "known failure" or "possibly fixed" | **gate FAIL** (same pressure as MUST_PASS) |
| **NA** | hardware-permanent (no upgrade will change it) | **no** | `na` + reason | neutral |
| *(unlisted)* | not in policy | **not run, not in report** | — | — |

*A feature that's absent on this hardware is `NA` (not run); flip it to `MUST_PASS`
when firmware adds it. There is no "optional, skip-is-fine" class — for an
acceptance gate that middle ground is exactly the ambiguity this taxonomy removes.*

---

## 8. Result schema (JSON — hand-emitted)

Each rank/node emits its own per-rank results as a **key=value file** (`rank=N`
header + `@test` record blocks — *not* JSON, so nothing has to *parse*
JSON); the **Reporter** merges nodes, reconciles the cross-rank verdict (§7.3),
joins each test with its registry metadata + classification, and **synthesizes**
rows the worker never emitted (`NA`; crash/timeout EXEC children — §6.3). The final
**`report.json` is the only JSON, and it is hand-emitted** (no third-party JSON
library): the structure is flat and we own it, so a ~60-line writer (with a
string-escaper + `std::to_chars` for locale-independent numbers) suffices.

**Always emit the full report — no `--verbose`.** `report.json` is a
machine artifact, so it *always* carries the full per-test detail (see the block
below). The diagnostic fields must be present on first failure (CI re-runs are
unreliable), and a lean view is trivially derived from a full report (`jq`), never
the reverse. `feature_tags`/`gates_evaluated`/`sub_tests` are **not emitted** (no
consumer yet, and the registry does not carry that metadata — §6). The `perf` map
**is** emitted (populated under `--kind perf|all`, `{}` otherwise — §9).

**`environment` is per node**: a heterogeneous cluster still yields a usable
report, with a top-level `homogeneous` rollup. `mode` is `gate` normally or
`discovery` when no policy file exists (§7.2); `filtered` is `true` when a
selection filter narrowed the run.

```jsonc
{
  "schema_version": "1.2",
  "harness": { "version": "0.1.0-phase1", "git_sha": "9133e10-dirty" },
                                                // build-time harness version + git SHA;
                                                // git_sha is "unknown" for a build with
                                                // no .git (e.g. a source tarball)
  "mode": "gate",                              // gate | discovery
  "filtered": false,                           // true when --filter*/--kind narrowed the run
  "summary": { "passed": 0, "failed": 0, "skipped": 0, "not_applicable": 0 },
  "homogeneous": true,                         // rollup; false → inspect per-node env
  "nodes": [
    { "host": "node-a", "rank": 0,
      "environment": {                         // detected dimensions, per node
        "vendor": "thor", "rdma_prefix": "bnxt_re",
        "rocm_version": "7.0.1", "rocm_path": "/opt/rocm-7.0.1",
        "rocm_version_source": "runtime_info",                  // runtime_info | rocm_path | hip_runtime | opt_rocm | none
        "gpu_count": 8,
        "kernel": "5.15.0", "os": "Ubuntu 22.04",
        "gpu_model": "…", "driver_version": "…",
        "symbols": { "libionic": false, "hsa_amd_portable_export_dmabuf": true },
        "rdma_devices": [
          { "device": "bnxt_re0", "netdev": "…", "fw_ver": "…", "board_id": "…",
            "node_guid": "…", "gid_index": 3, "gid_type": "RoCEv2",
            "vendor_id": null, "vendor_part_id": null }
        ],
        "gpu_devices": [ { "ordinal": 0, "bdf": "…", "numa_node": 0, "arch": "gfx…" } ],
        "gpu_nic_affinity": [ { "gpu": 0, "nearest_nic": "bnxt_re0" } ]
      } },
    { "host": "node-b", "rank": 1, "environment": { "...": "..." } }
  ],
  "tests": [
    {
      "id": "vat_device_enum", "category": "verbs/foundation",
      "class": "must_pass", "kind": "functional", "topology": "single",
      "status": "pass",            // POST reduction+classification: pass | fail | skip | na
      "reduced_from": { "0": "pass" },   // rank-level inputs, ONLY pass|fail|skip|none (§7.3)
      "expected_status": null,           // KNOWN_FAILURE only: fail|pass|skip|none (default fail)
      "reason": null,
      "when_matched": null,              // the firing `when` predicate, if any
      "ticket": null,
      "synth": false,                    // true if Reporter-synthesized (NA / crash / timeout)
      "duration_ms": 12,
      "perf": {},                        // metric key→value map; populated under --kind perf|all (§9)
      "gate_fail": false,                // this test's contribution to the gate verdict
      "flagged": false,                  // e.g. KNOWN_FAILURE "possibly fixed"
      "note": null
    }
  ]
}
```

---

## 9. Performance measurement (implemented, opt-in)

Performance is *secondary* to the functional gate but is **implemented** and folded
into the same test bodies, with **no new dependency** (Google Benchmark would
violate the "no major SW-stack requirements" rule). It is off by default:

- **Opt-in dimension.** `--kind func` (default) runs functional only; `--kind perf`
  or `--kind all` turns measurement **on** globally. `Kind::Performance` exists in
  the type system but no test registers with it — measurement is a global mode, not
  a per-test kind. The flag is forwarded to isolated `--exec` children.
- **Two measurement mechanisms** (`common/Perf.hpp`):
  - **Region timing — `PerfLoop`.** An iterable that runs warmup + N iterations of
    a region; when perf is on it emits per-metric rows, otherwise it runs a single
    functional pass. The iterator yields a `Pass` helper (non-trivial destructor)
    so loop bodies need no unused-variable annotations.
  - **Per-call latency — `PerfProbe`.** A thread-local scope that captures the
    latency of each `HIP_CHECK`-wrapped call; the captured samples are folded into
    the returned `TestResult` (`run_with_probe`).
- **Metric encoding (as-built).** Keys are `<metric>.<stat>[.<size>]`, e.g.
  `h2d_copy.median.256MiB`. Emitted stats are **`median`, `min`, `max`, `avg`**,
  plus companion `<metric>.unit`, `<metric>.iters`, and a raw
  `<metric>.samples.<unit>` array; `<size>` is humanized (`64KiB`, `256MiB`).
  Values are fixed-decimal strings. They land in the per-test `perf{}` map (§8).
- **Isolation caveat.** One-fork-per-test (§6.3) means no warm process state across
  tests — warmup happens *within* a single `PerfLoop`, not across tests.
- **Not built:** baseline/tolerance-based **regression gating** — perf is
  measure-and-record only; metrics never fail the gate today (§13).

---

## 10. Build & packaging

- **Standalone CMake project**, buildable on a bare node — **no RCCL include path, ever**.
- Hard deps: `-ldl`, `pthread` only. **`libibverbs` and `libamdhip64` are NOT
  linked** — they (and all optional symbols) are resolved at runtime via `dlopen`
  through `wrapper/` shims, using vendored headers
  (`vendor/rocm-headers`, `vendor/rdma-core-headers`). **No GoogleTest**
  and **no third-party JSON library** — zero third-party code beyond
  libc/system libs. Verbs and HIP test sources are compiled behind CMake options
  (`ACCEPT_WITH_IBVERBS`, `ACCEPT_WITH_HIP`); MPI is not supported.
- **Product:** the single `hva` binary + `policy/*.csv`. Flat.
  A separate `unit_tests` binary (+ CTest target) exercises the harness internals.
- **Language standard:** C++20.

**No GoogleTest / no third-party JSON.** Both were analyzed and
declined for this suite. GoogleTest's in-process model fights the
out-of-process regression signal and its benefits are ~50 LOC to replicate; the
JSON library was build-time-only but is removed for **auditability/minimalism** of
a foundational acceptance tool since we own every format (we *emit* JSON by hand —
trivial — and read a non-JSON policy CSV — ~15-line reader — so we never *parse*
arbitrary JSON). Net: the whole binary is auditable, with ~100 LOC of
writer/reader covered by the self-test.

> *Implementation status:* **done & node-validated** — `nlohmann` and
> `classification.json` are removed; the binary reads per-vendor `policy/*.csv`,
> emits `report.json` by hand, and uses key=value internal result files.

### Source layout (repo, as-built)

```
hip-verbs-acceptance/
  README.md               # entry point: build + run
  CMakeLists.txt
  main.cc                 # CLI, mode dispatch (LAUNCH / WORKER / EXEC)
  docs/                   # ARCHITECTURE.md ADDING_A_TEST.md
                          # POLICY_REFERENCE.md NEW_PLATFORM.md
  policy/*.csv            # per-vendor / per-platform classification,
                          # e.g. connectx.csv, thor.csv, ionic.csv, mi350_thor.csv
  common/                 # Types, TestRegistry, Runner (isolation+watchdog),
                          # Bootstrap + SocketBootstrap, Capability probe,
                          # Policy (CSV) loader, Reporter (JSON), Perf
  wrapper/                # dlopen/dlsym shims for libibverbs / libamdhip64
  vendor/                 # vendored rocm-headers / rdma-core-headers
  verbs/                  # vat_*  — libibverbs tests, incl. mixed HIP+verbs
                          #          (hip_mixed.cc, hip_coherence.cc, …)
  hip/                    # HIP tests (hip/utils/ helpers, perf memcpy)
  synthetic/  tests/  scripts/   # harness self-validation + unit tests
```

*Source is grouped by domain (build-dependency isolation, scale, team ownership);
the deployed product is flat (one binary + `policy/*.csv`). There is no `mixed/`
directory — mixed HIP+verbs tests live under `verbs/`.*

### 10.1 Deployment & privileges

- **Binary placement (as-built).** The launcher assumes the **same path is valid
  on every node** (shared FS or a pre-staged identical binary): each worker runs
  `--remote-binary` (default: the launcher's own `/proc/self/exe` path). Automatic
  **scp-based deploy** to non-shared-FS clusters and shared-FS auto-detection are
  **not implemented** (§13).
- **ABI mismatch.** Because workers run the launcher's binary path, the launcher
  must run on a host whose runtime is compatible with the target nodes; there is no
  automatic build-triple/glibc verification yet.
- **Environment (as-built).** ssh workers do **not** otherwise share the
  launcher's environment — they start with the remote login environment, which
  can point at a different (or no) ROCm. `--nodes` now forwards a fixed list of
  launcher environment variables to each worker by default
  (`LD_LIBRARY_PATH,ROCM_PATH,PATH,HIP_VISIBLE_DEVICES,ROCR_VISIBLE_DEVICES,HSA_*,ACCEPT_*`;
  `--forward-env` overrides the list, `--no-forward-env` disables it). This is a
  **behaviour change**: previously, ssh workers always ran with the remote
  login environment only. `build_remote_command` prefixes the command with
  `env 'K=V' ...` (each pair single-quoted as one token) when the resolved list
  is non-empty; see `common/SshLauncher.cpp`. Applies to the ssh launcher only:
  `--remote-launcher srun` passes the caller's whole environment itself
  (`--export=ALL`), adds no `env` prefix, and rejects `--forward-env` /
  `--no-forward-env` (exit 2).
- **Privileges & destructive tests (not implemented).** Link-down/`nicctl`-style
  destructive tests and an `--allow-destructive` gate are **planned, not built**
  (§13). The suite currently issues no destructive operations.

---

## 11. Command-line interface

No startup *config* file (the external artifact is the `policy/*.csv` data, not run
configuration). Deployment is driven entirely by CLI options, viable because the
environment is auto-detected.

```
hva                            # single-node loopback, all func tests
hva --nodes h1,h2              # multi-node launcher (ssh or srun)
           [--remote-launcher ssh|srun]     # default: ssh
           [--ssh-user <name>]              # default: ~/.ssh/config / current user
           [--port <n>] [--remote-binary <path>] [--launch-timeout <sec>]
           [--forward-env <list>] [--no-forward-env]  # default list forwards
                                            # LD_LIBRARY_PATH,ROCM_PATH,PATH,
                                            # HIP_VISIBLE_DEVICES,ROCR_VISIBLE_DEVICES,
                                            # HSA_*,ACCEPT_* (ssh launcher only)
           [--kind func|perf|all]           # default: func (perf/all turn on metrics)
           [--filter '<fnmatch glob>']      # glob on test id (fnmatch), e.g. 'vat_*'
           [--filter-class <class>]         # select by policy class
           [--filter-category <glob>]       # select by ACCEPT_TEST category field
           [--policy ./policy/thor.csv]     # direct path to policy CSV; auto-selected from
                                            # <binary_dir>/policy/<gpu>_<nic>.csv → <nic>.csv;
                                            # missing → WARN + all MUST_PASS (gate active)
           [--vendor <name>]                # override vendor auto-detection
           [--report ./report.json]         # always the full report (no --verbose)
           [--results-out <path|->]         # worker result file (- = stdout)
           [--isolate all|none]             # process-isolation scope (default: all)
           [--gid <index>] [--device <name>]  # override GID / RDMA device selection
           [--hide-na]                      # suppress NA rows from console
           [--lint]                         # list unlisted tests, exit non-zero, no run
           [--quiet]                        # suppress the per-test console stream
           [--local-ranks <n>]              # ranks per host (single-host multi-rank)
```

Internal modes (used by the launcher, not typed by hand):
`--worker --rank <r> --size <n> --rendezvous <host:port>` and
`--exec <test_id>`. `--filter` is an **fnmatch glob** on the test id (no
GoogleTest). **Not implemented:** `--verbose`, `--dump-requirements`,
`--allow-destructive`. Run `hva --help` for the current set.

---

## 12. Phasing (maps to spec Scope & Tasks)

| Phase | Deliverable | Spec step | State |
|---|---|---|---|
| 1 | Harness end-to-end (registry, EXEC isolation+watchdog, policy/classification, Reporter §7.3, SocketBootstrap size≤2, CMake) proven by a synthetic self-test covering every failure mode + a **2-rank smoke validated on two real nodes**. | 3 | **✅ done** |
| 1b | Real ssh **`--nodes` multi-node launcher + merge** into one gated `report.json`. | 3 | **✅ done** |
| 2 | `verbs/` functional — `vat_common` helpers + the `vat_*` tests; capability/vendor probe. | 4 | **✅ done** |
| 3 | `hip/` functional | 6 | **✅ done** |
| 4 | mixed HIP+verbs functional (under `verbs/`) | 8 | **✅ done** |
| 5 | Per-test **performance measurement** (`--kind perf\|all`, §9) | 5, 7, 9 | **✅ done** |
| 6 | Classification enforcement + cross-vendor matrix (`--dump-requirements`) automation | 1, 2, 10 | ⬜ |
| 7 | CI integration on public runners; reproduce known real failures | 11 | ◧ in progress |
| 8 | Performance **regression gating** (baselines + tolerance) | 5, 7, 9 | ⬜ |

**The harness was built and validated end-to-end (launch → socket bootstrap →
worker → per-rank result → Reporter reconciliation) on real hardware. The result
channel is the harness's own key=value `TestResult` file, JSON only in the merged
`report.json` (see §6.3 / top status note).**

---

## 13. Open items / future work

**Phase 1b — DONE & node-validated:**

- **Real ssh `--nodes h1,h2` launcher + report merge** is built: ssh-launches one
  worker/node (rank/size/rendezvous = node0:port), captures each rank's results on
  its ssh child's **stdout** (`--results-out -`; no shared-FS dependency), bounded
  (`ConnectTimeout` + `--launch-timeout`), Reporter merges → one gated
  `report.json`; exit 4 on deploy/unreachable/timeout. Validated across two nodes.
  *Deferred:* scp-based deploy for non-shared-FS clusters, and a **portable binary**
  (static libstdc++ / build on oldest target) so the launcher can run on an older
  **head** node — today it must run on a host with a compatible runtime that can
  ssh the targets (we ran it on a compute node; compute→compute ssh works).

**Validated-on-hardware lessons (now in code):**

- **Cross-host > localhost testing is mandatory.** Localhost rendezvous (numeric
  `127.0.0.1`) masked two real socket bugs, both fixed: the listener must bind
  `0.0.0.0` (a node's own FQDN can resolve to `127.0.1.1`), and the connector must
  resolve hostnames via `getaddrinfo` (not just `inet_pton`). See §4 note.
- **Orchestration:** double-hop ssh collapses quoting (pipe scripts via stdin);
  `ssh` eats stdin on fan-out (use `ssh -n`); NFS needs close-to-open settle
  (`sync; sleep`) before reading a peer-written file. The `--nodes` launcher must
  bake these in.

**Open dependencies (block specific functionality, not the harness):**

- **New NIC device-name prefixes.** The vendor key is derived from the ibverbs
  device-name prefix (`mlx5_`→connectx, `bnxt_re`→thor, `ionic_`/`rdma_`→ionic,
  `Capability.cpp`). A NIC whose prefix is not yet mapped runs in **discovery
  mode** (all `MUST_PASS`, §7.2). *Action:* confirm the prefix and add the rule.
- **gid_index selection is best-effort across vendors** (§5) — needs validation
  on each target NIC; the `--gid <index>` override is the escape hatch.

**Deferred (interfaces/seams in place):**

- **scp-based deploy for non-shared-FS clusters** and shared-FS auto-detection
  (§10.1) — today the binary path must be valid on every node.
- **Privileged / destructive tests** (link-down, `nicctl`) + an
  `--allow-destructive` gate (§10.1) — not yet built.
- **`--dump-requirements`** cross-vendor matrix rendering and `--verbose`.
- **Rank→GPU binding** for single-host multi-rank runs (§5).

- True **N-rank `SocketBootstrap`** (allgather/barrier for size > 2) — current
  scope is size ≤ 2, which covers the entire VAT set; pairs with the launcher
  work below.
- **Coordinated peer-failure exit** — *only if* the 3 pairwise tests
  (8.1/8.2/8.3) prove flaky or their hang-then-kill failure-path latency becomes a
  measured CI annoyance. If so, add **one** narrow primitive — a
  `bootstrap.abort_barrier()` a failing rank signals and a waiting peer polls with
  a short timeout — **not** a mirror of the MPI `ASSERT_BOOTSTRAP_*` macro family.
  This buys the graceful-exit benefit for a fraction of the cost and without the
  size==1 no-op and §7.3 double-count footguns (see §4 non-goal, §7.3 rule 4).
- `mpirun` launcher plugin (launcher/bootstrap split already admits it).
  *`srun` is implemented* (`--remote-launcher srun`) — one
  task per allocated node via `srun --nodes=1 --ntasks=1 --nodelist=<n>
  --overlap --export=ALL`, for clusters where direct ssh between compute nodes is blocked.
  Requires Slurm >= 20.11 (`--overlap`) and `SLURM_JOB_ID` set (i.e. running
  inside an allocation).
- `fw_version`-ranged classification entries (firmware-aware gating).
- Performance baselines + tolerance-based regression gating (§9).
- Multi-NIC-per-node and full heterogeneous-cluster support (current assumption:
  homogeneous; per-node descriptors recorded and mismatch flagged, not fatal).

**Robustness follow-ups (review):**

- **Between-test health probe** — after an isolated test crashes/SIGKILLs,
  verify the device/port is usable before the next test, so leaked RDMA/GPU
  resources or a wedged port surface as a clear error rather than spurious
  downstream failures. The fork-per-isolated-test model (§6.3) already gives a
  fresh device open; the probe closes the remaining gap.
- **Distributed-level watchdog** — beyond the per-test watchdog, the launcher
  needs a peer-death/hung-rank timeout so a worker that never reaches rendezvous
  (or dies mid-run) fails the run cleanly instead of hanging the whole job.
- **Observability** — per-rank logs collected to the launcher's `--report`
  sibling dir; environment descriptor captured verbatim for reproducibility.
