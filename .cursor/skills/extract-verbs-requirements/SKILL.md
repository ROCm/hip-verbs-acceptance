---
name: extract-verbs-requirements
description: Tell me which Verbs (libibverbs / rdma-cm / mlx5dv) acceptance tests are missing and need to be added. Analyzes a source tree (RCCL/NCCL/NIXL/etc.) for how it uses the RDMA verbs API, compares against an embedded knowledge base of required acceptance scenarios AND against the verbs tests that already exist (rccl test/transport NetIbMPI + VerbsMPI + this repo's tests), then emits a ready punch-list (PROPOSED_TESTS.md + proposed_tests.json) of tests to add — each with name, sequence, class, gate, verify, provenance. Use when someone wants to refresh the Verbs Test Plan from RCCL/NCCL code, find newly-used ibv_*/rdma_*/mlx5dv_* APIs or known-issue references not yet in the plan, or know which required verbs scenarios are not yet covered. Verbs counterpart of extract-hip-requirements. Self-contained: needs no external test plan.
---

# extract-verbs-requirements

**What it gives you:** run it, get `PROPOSED_TESTS.md` — a list of Verbs acceptance
tests still missing, each described well enough to implement. It (1) reads how the
source uses libibverbs/rdma-cm/mlx5dv, (2) checks an embedded catalog of required
acceptance scenarios (`knowledge/verbs_acceptance_kb.json`), and (3) subtracts
what is already tested in the verbs test corpus (NetIbMPI, VerbsMPI, the local repo).

Verbs sibling of `extract-hip-requirements`; same architecture, different domain.
Static, read-only: never builds, never needs a GPU/NIC. Needs only `bash` +
`python3` (no `jq`).

## Quickstart

Run it **from inside the repo you want analyzed** (scans CWD by default):
```
cd <your-repo>
/extract-verbs-requirements                          # scan the current directory
/extract-verbs-requirements --plan ./verbs_plan.md   # cross-ref an existing plan
/extract-verbs-requirements /path/to/repo            # explicit path
```
Open the output dir → read **PROPOSED_TESTS.md** (`## TO ADD (PROPOSE)` is the work
list); `proposed_tests.json` is the agent-readable form.

### What to report in chat (keep it short)

`extract.sh` ends with a `RESULT:` block — relay just that: counts (to-add /
unmodeled), the created file paths, and the next step (review UNMODELED, offer to
add KB scenarios). Don't paste full tables into chat; they live in the files. Each
UNMODELED symbol in PROPOSED_TESTS.md carries a short code excerpt (how it is called
in the source) so the name is understandable.

### Search / paths

No hardcoded user paths. Default source = CWD; named repos searched via
`$HIPREQ_SEARCH_ROOTS` → CWD/ancestors → enclosing git root + its `projects/`.
Output written under CWD (`./verbs-req-<repo>`).

## Key facts

- **Verbs API shape:** `ibv_*`, `rdma_*`, `mlx5dv_*` (lowercase, library-prefixed).
  Unlike HIP, there is no CUDA↔HIP duplication, so symbols are already canonical —
  no normalization step.
- **Coverage corpus = the verbs test suites**, not hip-tests: `rccl
  test/transport/NetIbMPI`, `test/transport/VerbsMPI`, the rest of `rccl/test`, and
  the repo the skill lives in. (Drop the skill into a fresh verbs-acceptance test
  repo → it reports what is still missing locally; tests flip to COVERED as added.)
- **Coverage is scenario-level**, not API-level: a scenario is COVERED only when a
  test implementing it exists (matched by `match_names`), not because its APIs
  appear somewhere. So a shared primitive (e.g. `ibv_post_send`) being used in one
  test does not mark every RDMA scenario covered.
- Status per item: **PROPOSE** (add), **COVERED** (test exists, where), **IN-PLAN**
  (named in the cross-ref plan), **NEW**/**UNMODELED** (used but no KB scenario yet).

## How to run

```
bash scripts/extract.sh --source <name|path> [--plan <plan.md>] [--corpus <root[:tag]>] [--out <dir>]
```
Outputs in `<dir>`: `api_surface.tsv`, `gates_issues.tsv`, `call_sequences.txt`
(scenario leads), `stress_leads.tsv` (API-in-loop), `coverage_corpus.tsv`,
`joined.tsv` (status per symbol), `PROPOSED_TESTS.md` + `proposed_tests.json`,
`SUMMARY.md`.

## Procedure (when invoked)

1. Pass the user's source to `extract.sh --source`; it auto-resolves repo + verbs
   test corpus. Add `--plan <verbs_plan.md>` to mark already-listed scenarios IN-PLAN.
2. Read `PROPOSED_TESTS.md` + `call_sequences.txt` + `gates_issues.tsv`.
3. For each PROPOSE (and for sequences/known-issues not in the KB), author a plan
   row in the user's table format `Test | Description | Class | Gate`, reading the
   cited `file:line` to ground the sequence. Scripts find leads; the model writes
   the scenario prose + class (skip-vs-fail is a judgement call).
4. Emit a delta for review; UNMODELED symbols are candidates to extend the KB.

## Maintaining the KB

`knowledge/verbs_acceptance_kb.json` is the standard (32 scenarios derived from the
Verbs Acceptance plan: foundation, connection, host data path, GPU/DMABUF, GPU-mem
correctness, recovery/error, cross-node, GDR handoff). Each entry: `id, title, area,
class, gate, signal_apis[], sequence[], known_issue, description, verify,
provenance_hint, match_names[]`. When a scan surfaces a used `ibv_*/mlx5dv_*` symbol
with no KB scenario (joined.tsv NEW / UNMODELED), add an entry.

## Files

- `scripts/extract.sh` — orchestrator (run this).
- `scripts/resolve_source.sh` — repo name/alias/path → source dir.
- `scripts/scan_api_surface.sh` — verbs/rdma/mlx5dv call enumeration.
- `scripts/scan_call_sequences.sh` — per-function ordered call chains (scenarios).
- `scripts/scan_stress_leads.sh` — API-called-inside-a-loop (stress candidates).
- `scripts/scan_gates_and_issues.sh` — version gates / JIRA ids / probes.
- `scripts/build_coverage_corpus.sh` — index the verbs test suites.
- `scripts/normalize_cuda_to_hip.sh` — passthrough (verbs needs no normalization).
- `scripts/propose_tests.sh` — KB × usage × coverage → PROPOSED_TESTS.md + json.
- `knowledge/verbs_acceptance_kb.json` — embedded requirements standard.
