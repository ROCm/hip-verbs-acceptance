---
name: extract-hip-requirements
description: Tell me which HIP acceptance tests are missing and need to be added. Analyzes a source tree (RCCL/NCCL/rocSHMEM/NIXL/Mooncake) for how it uses the HIP/CUDA/HSA APIs, compares against an embedded knowledge base of required acceptance scenarios AND against the tests that already exist (this repo's tests + hip-tests + others), then emits a ready punch-list (PROPOSED_TESTS.md + proposed_tests.json) of tests to add — each with name, sequence, class, gate, verify, provenance and an implementation hint precise enough for an agent to write the test. Use when someone clones a HIP/Verbs acceptance test repo and asks "what tests should I add", wants to refresh requirements from new RCCL/NCCL code, or wants to know which required HIP scenarios are not yet covered. Self-contained: needs no external test plan.
---

# extract-hip-requirements

**What it gives you:** run it, get `PROPOSED_TESTS.md` — a list of HIP acceptance
tests you still need to add, each described well enough to implement. It figures
this out by (1) reading how the analyzed source uses HIP, (2) checking an embedded
catalog of required acceptance scenarios (`knowledge/hip_acceptance_kb.json`), and
(3) subtracting what is already tested anywhere it can see — including the test
repo the skill is sitting in.

Static, read-only: never builds, never needs a GPU. Runs wherever source is
checked out. Needs only `bash` + `python3` (no `jq`).

## Quickstart (zero context required)

Run it **from inside the repo you want analyzed** — it scans the current directory
by default:
```
cd <your-repo>
/extract-hip-requirements
```
Then open the printed output dir and read **PROPOSED_TESTS.md**. The `## TO ADD
(PROPOSE)` section is your work list; `proposed_tests.json` is the same data for an
agent to consume. You don't need to know about any test plan or repo layout; the
skill resolves sources and coverage from where it is run.

Common variants:
```
/extract-hip-requirements                 # scan the current directory (default)
/extract-hip-requirements rocshmem        # a sibling repo by name (searched near CWD)
/extract-hip-requirements /path/to/repo   # by explicit path
```

### The two-file output (start here)

- **PROPOSED_TESTS.md** — human + agent readable. Per proposed test: id, title,
  class (MUST-PASS/CONDITIONAL/REGRESSION), gate, sequence of primitives, what to
  verify, signal APIs, provenance (where in the source it comes from), impl hint.
- **proposed_tests.json** — machine form. An agent reads `.items[] | select(.status=="PROPOSE")`
  and writes each test. Status is one of PROPOSE (add it), COVERED (already tested,
  with where), N/A (this source doesn't use those APIs).

### Coverage is SCENARIO-level, not API-level (important)

A requirement is COVERED only when a test that implements *that scenario* exists —
matched by the requirement's `match_names` (its title/id) appearing in a corpus
test file's `TEST(...)` name, a test filename, or a plan/spec doc's text. **API
overlap is NOT coverage.** If a scenario's APIs are unit-tested elsewhere (e.g.
hip-tests tests `hipMemMap`) but no test implements the *chain* (e.g.
`Vmm_Map_NoDeadlock_KernelInFlight`), it still PROPOSEs, and adds a "reuse hint"
pointing at the unit tests. This is why two scenarios that share APIs (in-process
vs real cross-process IPC) are kept distinct.

### Unmodeled symbols (the discovery channel)

The curated KB can be incomplete. So the proposer also reports **UNMODELED** —
canonical HIP symbols the analyzed source uses that no KB requirement lists as a
`signal_api` and that aren't already covered. These are candidates for NEW KB
entries (it restores the breadth of a raw symbol scan on top of curated scenarios).
`proposed_tests.json` carries them in `.unmodeled_symbols`; when you see a real
scenario there, add a KB entry for it.

### Where "already covered" comes from (and the new-repo workflow)

Coverage = the union of: **the test repo this skill lives in** (its `tests/` or
`test/` dir — so a freshly cloned acceptance repo knows what it already has),
hip-tests, rccl/test, rccl-tests, rocshmem/tests, and any `--corpus <root[:tag]>`.
When you drop this skill into a new acceptance-test repo and run it, every required
scenario you have not implemented yet shows up as PROPOSE; as you add tests, they
flip to COVERED on the next run. No test plan needed — the knowledge base is the
standard, the repo is the progress tracker.

Background (for maintainers): HIP acceptance requirements, Step 1.

## What to report in chat (keep it short)

`extract.sh` prints a `RESULT:` block at the end. Relay just that to the user:
- counts (how many to add / how many unmodeled),
- the paths to the created files (PROPOSED_TESTS.md, proposed_tests.json, SUMMARY.md),
- the next step (review the UNMODELED section; for each real one, offer to add a KB
  scenario, which flips it from UNMODELED to PROPOSE).

Do NOT paste the full tables into chat — they live in the files. The UNMODELED
section of PROPOSED_TESTS.md now includes a short code excerpt per symbol (how it is
called in the source) so the names are understandable without guessing.

## Why this exists / key facts

- **RCCL keeps CUDA names.** RCCL is hipified NCCL; `src/` calls `cuMemCreate`,
  `cuLaunchKernel`, `cudaStreamCreateWithFlags`, and maps them to HIP via
  `src/include/cudawrap.h` + hipify. So RCCL and NCCL share one symbol language;
  grep both with the same patterns and normalize to a canonical **HIP** name on
  input. Canonical = HIP because that is what the standalone acceptance binary
  calls and what `projects/hip-tests` is indexed by.
- **"What already exists for HIP" is repo-wide, not just hip-tests.** Coverage is
  measured against a CORPUS of every test suite in the monorepo (hip-tests +
  `rccl/test` + `rccl-tests` + `rocshmem/tests` + any `--corpus` you add), each with
  its own provenance tag, PLUS a cross-reference of the existing HIP Test Plan
  (`--plan`). Every candidate gets a 3-way status:
  - **COVERED** — a unit test for this symbol exists somewhere; `covered_in` says
    which suite(s) and `covered_paths` the file(s). Reuse target. (But a chain or
    stress angle on top may still be NEW — judge from call_sequences/stress.)
  - **IN-PLAN** — already named in the HIP Test Plan .md. Skip unless a new
    sequence/stress angle appears.
  - **NEW** — not in any corpus and not in the plan. Acceptance must build it.
- Acceptance is NOT duplicating unit tests. It is (A) RCCL's *sequences* of
  primitives, (B) known-regression repros (ROCM-2550/25387/2696, version-gated CE
  batch copy, …), (C) **stress/scale** variants where production loops an API,
  (D) version-as-a-dimension + symbol detection.
- **RCCL uses no `hipIpc*`** — cross-rank sharing is VMM POSIX-fd export/import.
  IPC-handle requirements come from secondary sources (rocSHMEM/NIXL/Mooncake),
  not RCCL.

## How to invoke

`/extract-hip-requirements [source]`. **Zero arguments works** — it defaults to the
current working directory and auto-resolves everything else. `source` may be a repo
**name/alias** or a **path**; the skill locates the repo and its source subdir itself.

```
/extract-hip-requirements                 # scan CWD (default)
/extract-hip-requirements rocshmem        # by alias (searched near CWD)
/extract-hip-requirements nccl            # by alias (if checked out nearby)
/extract-hip-requirements /path/to/repo   # by path
/extract-hip-requirements mooncake --no-hip-tests   # secondary source, skip coverage join
```

Known aliases: `rccl`, `nccl`, `rocshmem`, `nixl`, `mooncake`, `hip-tests`. Any
other token is treated as a repo directory name and searched for. Unknown names
fail loudly and print the discoverable repo list (`resolve_source.sh --list`).

### Autonomy / how it finds things (no hardcoded user paths)

`scripts/resolve_source.sh` does the locating, all relative to where you run it:
- **Default (no arg / `.` / `cwd`):** scans the current working directory. Run the
  skill inside a checkout and it analyzes that checkout — nothing else needed.
- **Search roots for named repos:** `$HIPREQ_SEARCH_ROOTS` (if set) → the CWD and a
  couple of ancestors → the enclosing git repo root and its `projects/` dir. No
  user-specific paths are baked in.
- **Source-subdir detection:** inside a repo it picks the dir that holds C/C++
  (`src/`, `library/src/`, `catch/`, `source/`, `lib/`, else the repo root) — adapts
  per repo without being told.
- **hip-tests reference:** auto-resolved near the source if present, else skipped
  (`--no-hip-tests` to force-skip). Coverage just degrades gracefully.
- **Out dir:** defaults to `./hip-req-<repo>` if `--out` omitted (written under CWD).

To analyze a repo elsewhere: pass its path as `source`, or
`export HIPREQ_SEARCH_ROOTS=/my/workspace` first.

### Dependencies

`bash` + `python3` only. No `jq`, no build, no GPU. (`python3` is used for the KB
matching and JSON output; it is present by default on essentially all dev machines.)

## Procedure (what I do when invoked)

1. **Resolve roots.** Pass the user's `source` (name/alias/path) straight to
   `extract.sh --source`; it resolves the repo, the source subdir, and the
   hip-tests reference autonomously (see Autonomy above). Only set `--hip-tests`
   or `--no-hip-tests` if the user asked, or if the source is not a HIP-runtime
   consumer. If a hipify CUDA2HIP map file is available, pass `--hipify-map` for
   authoritative normalization.

2. **Run the scan pipeline** (scripts/ next to this file):
   ```
   bash scripts/extract.sh --source <name|path> [--hip-tests <path>|--no-hip-tests] [--out <dir>]
   ```
   Useful flags: `--plan <test-plan.md>` to cross-ref the existing plan (marks
   IN-PLAN); `--corpus <root[:tag]>` to add another test suite to the coverage
   corpus; `--no-hip-tests` to skip coverage entirely (non-HIP-runtime repos).
   This produces in `<dir>`:
   - `api_surface.tsv` — every called API symbol (symbol+`(` only, so vars like
     `cudaDev` are excluded), family, call count, sample file:line locations.
   - `gates_issues.tsv` — version GATEs (HIP_VERSION/CUDART_VERSION/…), ISSUE ids
     (tracker IDs like `ABC-1234`), runtime PROBEs (`*Supported(`).
   - `call_sequences.txt` — **the scenario engine.** API calls in SOURCE ORDER,
     grouped per enclosing function, for every function using ≥3 distinct APIs.
     This is what turns "symbols" into "sequences": e.g. `ncclMemAlloc_impl` shows
     `cuMemGetAllocationGranularity → cuMemCreate → cuMemAddressReserve → cuMemMap
     → cuMemSetAccess` — a ready acceptance scenario with function-name provenance.
   - `stress_leads.tsv` — **the stress engine.** Every API called inside a loop
     (for/while) — candidate stress/scale tests (e.g. the 16× VMM-map ROCM-2550
     repro). If production loops an API, the plan should have a stress variant.
   - `coverage_corpus.tsv` — symbol → which test suite(s) already cover it (+path),
     across the whole monorepo.
   - `joined.tsv` — per API: canonical HIP name + status (COVERED/IN-PLAN/NEW) +
     `covered_in` (which suites) + paths.
   - `SUMMARY.md` — NEW table, covered-but-where table, stress-leads, gate/issue
     table, next-step note.

3. **Diff against the current plan.** Read `HipAcceptanceTests.md` §3/§4 plus
   `joined.tsv`, `gates_issues.tsv`, and `call_sequences.txt`. Two kinds of gap:
   - a (canonical-symbol | gate | issue) tuple not represented → candidate row;
   - a *sequence* in `call_sequences.txt` whose chain isn't captured as a scenario
     even if its individual symbols are COVERED in hip-tests (this is the whole
     point — hip-tests unit-tests symbols, acceptance tests the chain).

4. **Author plan rows.** For symbol/gate/issue candidates AND for sequence
   candidates, in the plan's table format `Test | Description | Class | Gate |
   Provenance`. **Always open the cited file:line region and read the actual code**
   before writing the Description — `call_sequences.txt` gives the lead (which
   calls, what order, which function); you confirm operands/loops/error-handling by
   reading. Do not invent a sequence the code doesn't show.
   - Description = the concrete RCCL sequence (operations + what to verify).
   - Class: REGRESSION if tied to a JIRA/known issue or a hang/timeout signal;
     CONDITIONAL if version/capability gated; else MUST-PASS.
   - Gate: from `gates_issues.tsv` (e.g. ROCm≥7.12 / backport range).
   - Provenance: `<repo> <file:line / function>` + COVERED `<hip-tests path>` or NEW.

   Note the division of labor: scripts find *leads* (symbols, order, gates);
   the model authors *scenarios* (the prose chain + class). C++ function bodies
   aren't reliably machine-parseable, so the final sequence is a read-and-judge
   step, not a pure grep.

5. **Emit** `requirements-delta-<YYYY-MM-DD>.md` — only the new/changed rows, with
   a one-line note per row on what changed. Hand to the user to review and merge
   into `HipAcceptanceTests.md`.

## Secondary sources

Same pipeline, different `--source`. For rocSHMEM/NIXL/Mooncake the goal is the
APIs RCCL does NOT exercise (notably real `hipIpc*` and symmetric-heap patterns),
which feed the HIP/Verbs mixed scenarios (epic Steps 8/9). rocSHMEM is already
native HIP/HSA, so normalization is a no-op there.

## Guardrails

- Read-only. Never build, never run on-device, never edit the plan automatically —
  always emit a delta for human review.
- The scripts do the mechanical scan; the *classification and prose* are authored
  by me from the cited code, because skip-vs-fail class and the scenario wording
  are judgement calls, not greppable facts.
- Unknown driver symbols are emitted with a `:CHECK_MAP` suffix and multicast with
  `:NO_HIP_EQUIV` — surface these for manual mapping rather than guessing.

## Files

- `scripts/extract.sh` — orchestrator (run this).
- `scripts/resolve_source.sh` — repo name/alias/path → source dir (`--list`, `--hip-tests`).
- `scripts/scan_api_surface.sh` — API call-site enumeration (which APIs).
- `scripts/scan_call_sequences.sh` — per-function ordered call chains (the scenarios).
- `scripts/scan_stress_leads.sh` — API-called-inside-a-loop sites (stress candidates).
- `scripts/scan_gates_and_issues.sh` — version gates / JIRA ids / probes.
- `scripts/build_coverage_corpus.sh` — index every monorepo test suite (multi-source coverage).
- `scripts/normalize_cuda_to_hip.sh` — CUDA→HIP canonicalization (`--dump` to inspect).
- `scripts/propose_tests.py` — KB × usage × coverage → PROPOSED_TESTS.md + proposed_tests.json.
- `knowledge/hip_acceptance_kb.json` — embedded requirements standard (curated scenarios,
  classes, known issues). Edit/extend this to teach the skill new requirements; it is the
  source of truth the proposals are generated from.

## Maintaining the knowledge base

The KB makes the skill smart without a test plan. Each entry: `id, title, area,
class, gate, signal_apis[], sequence[], known_issue, description, verify,
impl_hint, provenance_hint`. A requirement is PROPOSED when its `signal_apis` are
used by the analyzed source and none are already covered. To add a requirement,
append an entry; to retire one, remove it. When the code-scan surfaces a NEW symbol
or known-issue not represented by any KB entry (see joined.tsv status=NEW and
gates_issues.tsv), that is the signal to add a KB entry for it.
