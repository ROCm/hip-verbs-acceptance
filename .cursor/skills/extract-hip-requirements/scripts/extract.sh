#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# extract.sh -- orchestrator for the extract-hip-requirements skill.
#
# Runs the read-only static-analysis pipeline over one source tree and writes
# raw artifacts that the skill (Claude) then turns into a reviewed requirements
# delta. This script does the mechanical grepping/normalization/coverage join;
# the human-readable classification (MUST-PASS / CONDITIONAL / REGRESSION) and
# the prose Description are authored by the model from these artifacts.
#
# Usage:
#   extract.sh --source <src-root> [--hip-tests <root>] [--hipify-map <file>]
#              [--out <dir>]
#
# Examples:
#   extract.sh --source .../projects/rccl/src \
#              --hip-tests .../projects/hip-tests \
#              --out ./hip-req-out
#   extract.sh --source .../projects/rocshmem/src --out /tmp/rocshmem-req
#
# Outputs (in --out, default ./hip-req-out):
#   api_surface.tsv      symbol/family/count/locations
#   gates_issues.tsv     version gates, JIRA ids, capability probes
#   joined.tsv           per-API: hip-name + COVERED/NEW(+path) + nearby gate/issue
#   SUMMARY.md           quick human overview + next-step instructions
# Read-only w.r.t. the source trees.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --source accepts a repo NAME/alias (rccl, rocshmem, nixl, ...) OR a path; it is
# resolved to a concrete source dir by resolve_source.sh. Defaults to rccl.
# --hip-tests likewise accepts a path or auto-resolves when omitted/"auto".
SRC="." HIPTESTS="auto" HIPIFY_MAP="" OUT="" PLAN=""
NO_HIPTESTS=0
declare -a EXTRA_CORPORA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)      SRC="$2"; shift 2 ;;
    --hip-tests)   HIPTESTS="$2"; shift 2 ;;
    --no-hip-tests) NO_HIPTESTS=1; shift ;;
    --corpus)      EXTRA_CORPORA+=("$2"); shift 2 ;;   # extra test root[:tag]
    --plan)        PLAN="$2"; shift 2 ;;               # existing test-plan .md to cross-ref
    --hipify-map)  HIPIFY_MAP="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Resolve the source (name/alias/path -> concrete src dir).
SRC_RESOLVED="$(bash "$HERE/resolve_source.sh" "$SRC" 2>/dev/null)" || true
if [[ -z "$SRC_RESOLVED" || ! -d "$SRC_RESOLVED" ]]; then
  echo "extract: could not resolve --source '$SRC'." >&2
  echo "  Try a path, or one of these discoverable repos:" >&2
  bash "$HERE/resolve_source.sh" --list 2>/dev/null | sed 's/^/    /' >&2
  exit 2
fi
SRC="$SRC_RESOLVED"

# Resolve hip-tests reference unless disabled.
if [[ "$NO_HIPTESTS" -eq 1 ]]; then
  HIPTESTS=""
elif [[ "$HIPTESTS" == "auto" || -z "$HIPTESTS" ]]; then
  HIPTESTS="$(bash "$HERE/resolve_source.sh" --hip-tests 2>/dev/null)" || true
  [[ -n "$HIPTESTS" ]] && echo "extract: auto-resolved hip-tests -> $HIPTESTS" >&2 \
                       || echo "extract: hip-tests not found; coverage = all NEW" >&2
fi

# Default out dir = ./hip-req-<repo-basename>.
if [[ -z "$OUT" ]]; then
  base=$(basename "$(dirname "$SRC")")   # repo name above the src subdir
  [[ "$base" == "." || -z "$base" ]] && base=$(basename "$SRC")
  OUT="./hip-req-${base}"
fi
echo "extract: source=$SRC" >&2
echo "extract: out=$OUT" >&2
mkdir -p "$OUT"

echo "[1/8] scanning API surface of $SRC" >&2
bash "$HERE/scan_api_surface.sh" "$SRC" "$OUT/api_surface.tsv"

echo "[2/8] scanning version gates / known issues / probes" >&2
bash "$HERE/scan_gates_and_issues.sh" "$SRC" "$OUT/gates_issues.tsv"

echo "[3/8] extracting per-function call sequences (scenario leads)" >&2
bash "$HERE/scan_call_sequences.sh" "$SRC" "$OUT/call_sequences.txt" 3

echo "[4/8] finding API-in-loop stress leads" >&2
bash "$HERE/scan_stress_leads.sh" "$SRC" "$OUT/stress_leads.tsv" 6

echo "[5/8] building monorepo coverage corpus (what already tests HIP, everywhere)" >&2
# Coverage is NOT just projects/hip-tests. Index every test corpus so a symbol can
# be COVERED in hip-tests AND/OR rccl-test AND/OR rocshmem, with per-suite paths.
CORPUS="$OUT/coverage_corpus.tsv"
declare -a CORPUS_ARGS=()
[[ -n "$HIPTESTS" && -d "$HIPTESTS" ]] && CORPUS_ARGS+=("$HIPTESTS:hip-tests")
# Discover the OTHER monorepo test suites relative to the resolved rccl repo, so
# coverage reflects the whole repo (rccl/test, rccl-tests, rocshmem/tests), not
# only hip-tests. (build_coverage_corpus skips any root that doesn't exist.)
rcclsrc="$(bash "$HERE/resolve_source.sh" rccl 2>/dev/null)"
if [[ -n "$rcclsrc" ]]; then
  repo="$(dirname "$rcclsrc")"; proj="$(dirname "$repo")"
  [[ -d "$repo/test" ]]            && CORPUS_ARGS+=("$repo/test:rccl-test")
  [[ -d "$proj/rccl-tests" ]]      && CORPUS_ARGS+=("$proj/rccl-tests:rccl-tests")
  [[ -d "$proj/rocshmem/tests" ]]  && CORPUS_ARGS+=("$proj/rocshmem/tests:rocshmem-tests")
fi
# CRITICAL for the "drop the skill into a fresh test repo" workflow: the repo the
# skill itself lives in IS a coverage source -> "what tests are already added
# here". Walk up from the skill dir to a repo root (has .git or tests/) and index
# its tests as the 'local-repo' corpus. This makes a clean clone self-aware:
# running the skill says exactly which proposed tests are still missing locally.
localrepo="$HERE"
while [[ "$localrepo" != "/" && ! -d "$localrepo/.git" && ! -d "$localrepo/tests" && ! -d "$localrepo/test" ]]; do
  localrepo="$(dirname "$localrepo")"
done
if [[ "$localrepo" != "/" && "$localrepo" != "$HOME" ]]; then
  # Acceptance suites keep their tests in hip/ and verbs/ (ACCEPT_TEST macro), not
  # only tests/ — index those too so already-implemented acceptance tests count as
  # coverage (otherwise every KB scenario shows PROPOSE even when it exists here).
  for td in tests test hip verbs; do
    [[ -d "$localrepo/$td" ]] && CORPUS_ARGS+=("$localrepo/$td:local-repo")
  done
fi
for c in "${EXTRA_CORPORA[@]}"; do CORPUS_ARGS+=("$c"); done
bash "$HERE/build_coverage_corpus.sh" "$CORPUS" "${CORPUS_ARGS[@]}"

# In-plan symbols: pull every hip*/cu*/hsa_amd_* token mentioned in the plan .md.
PLAN_SYMS="$OUT/.plan_syms.txt"; : > "$PLAN_SYMS"
if [[ -n "$PLAN" && -f "$PLAN" ]]; then
  grep -oE '(cu[A-Z]|cuda[A-Z]|hip[A-Z]|hsa_amd_)[A-Za-z0-9_]+' "$PLAN" 2>/dev/null \
    | sort -u > "$PLAN_SYMS"
fi

# coverage lookup: returns "tags\tpaths" across ALL corpora (';'-joined), or empty.
cov_tags() {
  local s="${1%%:*}"
  [[ -f "$CORPUS" ]] || return 0
  awk -F'\t' -v s="$s" '$1==s{ tags=tags sep $2; paths=paths sep $3; sep=";" }
                        END{ if(tags!="") print tags"\t"paths }' "$CORPUS"
}
in_plan() { # hip OR original symbol mentioned in plan?
  [[ -s "$PLAN_SYMS" ]] || return 1
  grep -qxF "$1" "$PLAN_SYMS" || grep -qxF "$2" "$PLAN_SYMS"
}

echo "[6/8] normalizing CUDA->HIP and joining multi-source coverage" >&2
{
  printf 'cuda_or_hip_symbol\thip_canonical\tfamily\tcall_count\tstatus\tcovered_in\tcovered_paths\tsample_locations\n'
  tail -n +2 "$OUT/api_surface.tsv" | while IFS=$'\t' read -r sym family count locs; do
    [[ -z "$sym" ]] && continue
    hip=$(bash "$HERE/normalize_cuda_to_hip.sh" "$sym" "$HIPIFY_MAP")
    IFS=$'\t' read -r tags paths < <(cov_tags "$hip")
    if [[ -n "$tags" ]]; then status="COVERED"
    elif in_plan "$hip" "$sym"; then status="IN-PLAN"; tags="plan"
    else status="NEW"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$sym" "$hip" "$family" "$count" "$status" "${tags:-}" "${paths:-}" "$locs"
  done
} > "$OUT/joined.tsv"

echo "[7/8] proposing tests from the embedded knowledge base" >&2
KB="${HIPREQ_KB:-$HERE/../knowledge/hip_acceptance_kb.json}"
corpus_roots=""
for ca in "${CORPUS_ARGS[@]}"; do corpus_roots+="${ca%%:*}:"; done
[[ -n "$PLAN" && -f "$PLAN" ]] && corpus_roots+="$PLAN:"
corpus_roots="${corpus_roots%:}"
# Pure-Python proposer (no jq dependency). Passes --src so UNMODELED symbols get
# code-context excerpts.
if command -v python3 >/dev/null && [[ -f "$KB" ]]; then
  python3 "$HERE/propose_tests.py" --out "$OUT" --kb "$KB" \
    --joined "$OUT/joined.tsv" --corpus "$CORPUS" \
    --corpus-roots "$corpus_roots" --src "$SRC" || \
    echo "extract: propose_tests step failed (continuing)" >&2
else
  echo "extract: skipping proposals (need python3 + knowledge/$(basename "$KB"))" >&2
fi

echo "[8/8] writing SUMMARY.md" >&2
total=$(( $(wc -l < "$OUT/joined.tsv") - 1 ))
new_cnt=$(tail -n +2 "$OUT/joined.tsv"   | awk -F'\t' '$5=="NEW"'     | wc -l)
cov_cnt=$(tail -n +2 "$OUT/joined.tsv"   | awk -F'\t' '$5=="COVERED"' | wc -l)
plan_cnt=$(tail -n +2 "$OUT/joined.tsv"  | awk -F'\t' '$5=="IN-PLAN"' | wc -l)
gate_cnt=$(( $(wc -l < "$OUT/gates_issues.tsv") - 1 ))
stress_cnt=$(( $(wc -l < "$OUT/stress_leads.tsv") - 1 ))
# propose count without jq: count "status": "PROPOSE" in the JSON
prop_cnt=0
[[ -f "$OUT/proposed_tests.json" ]] && prop_cnt=$(grep -c '"status": "PROPOSE"' "$OUT/proposed_tests.json" 2>/dev/null || echo 0)
unmod_cnt=0
[[ -f "$OUT/proposed_tests.json" ]] && unmod_cnt=$(python3 -c "import json,sys;print(json.load(open('$OUT/proposed_tests.json'))['summary']['unmodeled'])" 2>/dev/null || echo 0)
{
  echo "# extract-hip-requirements -- summary"
  echo
  echo "## >>> START HERE: ${prop_cnt} test(s) to add"
  echo
  echo "See **PROPOSED_TESTS.md** for ready-to-implement acceptance tests (name,"
  echo "sequence, class, gate, verify, impl hint) and **proposed_tests.json** for the"
  echo "machine-readable version an agent can consume to write them. Everything below"
  echo "is the supporting raw analysis."
  echo
  echo "- source: \`$SRC\`"
  echo "- coverage corpora: \`$(tail -n +1 "$CORPUS" 2>/dev/null | cut -f2 | sort -u | paste -sd', ' -)\`"
  echo "- plan cross-ref: \`${PLAN:-<none>}\`"
  echo "- distinct API symbols called: **$total** — COVERED(somewhere): $cov_cnt, IN-PLAN: $plan_cnt, **NEW: $new_cnt**"
  echo "- gate/issue/probe signals: **$gate_cnt** · API-in-loop stress leads: **$stress_cnt**"
  echo
  echo "## Genuinely NEW symbols (not in ANY test corpus, not in the plan)"
  echo
  echo '| HIP symbol | family | calls | sample location |'
  echo '|---|---|---|---|'
  tail -n +2 "$OUT/joined.tsv" | awk -F'\t' '$5=="NEW"{print "| `"$2"` | "$3" | "$4" | "$8" |"}' | head -60
  echo
  echo "## Covered, but check WHERE (reuse target vs needs an acceptance-level chain)"
  echo
  echo '| HIP symbol | covered_in | calls |'
  echo '|---|---|---|'
  tail -n +2 "$OUT/joined.tsv" | awk -F'\t' '$5=="COVERED"{print "| `"$2"` | "$6" | "$4" |"}' | head -40
  echo
  echo "## Stress / scale leads (API called inside a loop -> candidate stress test)"
  echo
  echo '| api | loop @ | location |'
  echo '|---|---|---|'
  tail -n +2 "$OUT/stress_leads.tsv" | awk -F'\t' '{print "| `"$2"` | "$3" | "$1" |"}' | sort -u | head -40
  echo
  echo "## Known-issue / version-gate signals (-> CONDITIONAL or REGRESSION rows)"
  echo
  echo '| kind | token | location |'
  echo '|---|---|---|'
  tail -n +2 "$OUT/gates_issues.tsv" | awk -F'\t' '{print "| "$1" | `"$2"` | "$3" |"}' | sort -u | head -80
  echo
  echo "## Next step (model)"
  echo "Read joined.tsv (status NEW + COVERED-but-elsewhere), call_sequences.txt,"
  echo "stress_leads.tsv, gates_issues.tsv. Author plan rows for: (a) NEW symbols,"
  echo "(b) sequences whose chain isn't a scenario even if APIs are COVERED,"
  echo "(c) stress variants where production loops an API. Each row: Test |"
  echo "Description | Class | Gate | Provenance(file:line + covered_in or NEW)."
  echo "IN-PLAN rows are already captured — skip unless the chain/stress angle is new."
} > "$OUT/SUMMARY.md"

# Concise, chat-ready summary block (the model relays this verbatim).
{
  echo "RESULT:"
  echo "  source:    $SRC"
  echo "  to add (PROPOSE): $prop_cnt"
  echo "  unmodeled (need discussion): $unmod_cnt"
  echo "  files:"
  echo "    $OUT/PROPOSED_TESTS.md     (proposed tests + UNMODELED-with-code-context)"
  echo "    $OUT/proposed_tests.json   (machine-readable)"
  echo "    $OUT/SUMMARY.md            (full raw analysis)"
  echo "  next: implement each PROPOSE item with the acceptance-add-test skill, then"
  echo "        run + summarize; review UNMODELED and add a KB scenario for real ones."
} >&2
echo "extract: done -> $OUT" >&2
