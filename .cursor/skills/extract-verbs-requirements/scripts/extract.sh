#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# extract.sh -- orchestrator for the extract-verbs-requirements skill.
#
# Verbs counterpart of extract-hip-requirements. Read-only static analysis of how
# a source tree uses libibverbs / rdma-cm / mlx5dv, matched against an embedded KB
# of Verbs acceptance scenarios and the existing verbs test corpus, to propose the
# acceptance tests still missing.
#
# Usage:
#   extract.sh --source <name|path> [--plan <plan.md>] [--corpus <root[:tag]>] [--out <dir>]
# Read-only w.r.t. the source trees.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="." OUT="" PLAN=""
declare -a EXTRA_CORPORA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)  SRC="$2"; shift 2 ;;
    --corpus)  EXTRA_CORPORA+=("$2"); shift 2 ;;
    --plan)    PLAN="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SRC_RESOLVED="$(bash "$HERE/resolve_source.sh" "$SRC" 2>/dev/null)" || true
if [[ -z "$SRC_RESOLVED" || ! -d "$SRC_RESOLVED" ]]; then
  echo "extract: could not resolve --source '$SRC'." >&2
  bash "$HERE/resolve_source.sh" --list 2>/dev/null | sed 's/^/    /' >&2
  exit 2
fi
SRC="$SRC_RESOLVED"

if [[ -z "$OUT" ]]; then
  base=$(basename "$(dirname "$SRC")")
  [[ "$base" == "." || -z "$base" ]] && base=$(basename "$SRC")
  OUT="./verbs-req-${base}"
fi
echo "extract: source=$SRC" >&2
echo "extract: out=$OUT" >&2
mkdir -p "$OUT"

echo "[1/7] scanning verbs/rdma API surface of $SRC" >&2
bash "$HERE/scan_api_surface.sh" "$SRC" "$OUT/api_surface.tsv"

echo "[2/7] scanning version gates / known issues / probes" >&2
bash "$HERE/scan_gates_and_issues.sh" "$SRC" "$OUT/gates_issues.tsv"

echo "[3/7] extracting per-function call sequences (scenario leads)" >&2
bash "$HERE/scan_call_sequences.sh" "$SRC" "$OUT/call_sequences.txt" 3

echo "[4/7] finding API-in-loop stress leads" >&2
bash "$HERE/scan_stress_leads.sh" "$SRC" "$OUT/stress_leads.tsv" 6

echo "[5/7] building verbs coverage corpus (what already tests verbs)" >&2
# Coverage corpus = the verbs test suites in the monorepo. No hip-tests here
# (that is HIP-runtime, not verbs). Discover relative to the resolved rccl repo.
CORPUS="$OUT/coverage_corpus.tsv"
declare -a CORPUS_ARGS=()
rcclsrc="$(bash "$HERE/resolve_source.sh" rccl 2>/dev/null)"
if [[ -n "$rcclsrc" ]]; then
  repo="$(dirname "$rcclsrc")"            # .../projects/rccl
  [[ -d "$repo/test/transport/NetIbMPI" ]] && CORPUS_ARGS+=("$repo/test/transport/NetIbMPI:netib-mpi")
  [[ -d "$repo/test/transport/VerbsMPI" ]] && CORPUS_ARGS+=("$repo/test/transport/VerbsMPI:verbs-mpi")
  [[ -d "$repo/test" ]]                     && CORPUS_ARGS+=("$repo/test:rccl-test")
fi
# The repo the skill itself lives in (fresh verbs-acceptance test repo) -> local-repo.
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

# In-plan symbols: verbs/rdma/mlx5dv tokens mentioned in the plan .md.
PLAN_SYMS="$OUT/.plan_syms.txt"; : > "$PLAN_SYMS"
if [[ -n "$PLAN" && -f "$PLAN" ]]; then
  grep -oE '(ibv_[a-z0-9_]+|rdma_[a-z0-9_]+|mlx5dv_[a-z0-9_]+)' "$PLAN" 2>/dev/null \
    | sort -u > "$PLAN_SYMS"
fi

cov_tags() {
  local s="${1%%:*}"
  [[ -f "$CORPUS" ]] || return 0
  awk -F'\t' -v s="$s" '$1==s{ tags=tags sep $2; paths=paths sep $3; sep=";" }
                        END{ if(tags!="") print tags"\t"paths }' "$CORPUS"
}
in_plan() {
  [[ -s "$PLAN_SYMS" ]] || return 1
  grep -qxF "$1" "$PLAN_SYMS" || grep -qxF "$2" "$PLAN_SYMS"
}

echo "[6/7] joining multi-source coverage" >&2
{
  printf 'symbol\tcanonical\tfamily\tcall_count\tstatus\tcovered_in\tcovered_paths\tsample_locations\n'
  tail -n +2 "$OUT/api_surface.tsv" | while IFS=$'\t' read -r sym family count locs; do
    [[ -z "$sym" ]] && continue
    canon="$sym"   # verbs symbols are already canonical (no CUDA<->HIP mapping)
    IFS=$'\t' read -r tags paths < <(cov_tags "$canon")
    if [[ -n "$tags" ]]; then status="COVERED"
    elif in_plan "$canon" "$sym"; then status="IN-PLAN"; tags="plan"
    else status="NEW"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$sym" "$canon" "$family" "$count" "$status" "${tags:-}" "${paths:-}" "$locs"
  done
} > "$OUT/joined.tsv"

echo "[7/7] proposing tests from the embedded knowledge base" >&2
KB="${VERBSREQ_KB:-$HERE/../knowledge/verbs_acceptance_kb.json}"
corpus_roots=""
for ca in "${CORPUS_ARGS[@]}"; do corpus_roots+="${ca%%:*}:"; done
[[ -n "$PLAN" && -f "$PLAN" ]] && corpus_roots+="$PLAN:"
corpus_roots="${corpus_roots%:}"
# Pure-Python proposer (no jq). --src lets UNMODELED symbols get code-context.
if command -v python3 >/dev/null && [[ -f "$KB" ]]; then
  python3 "$HERE/propose_tests.py" --out "$OUT" --kb "$KB" \
    --joined "$OUT/joined.tsv" --corpus "$CORPUS" \
    --corpus-roots "$corpus_roots" --src "$SRC" || \
    echo "extract: propose_tests step failed (continuing)" >&2
else
  echo "extract: skipping proposals (need python3 + knowledge/$(basename "$KB"))" >&2
fi

# SUMMARY
total=$(( $(wc -l < "$OUT/joined.tsv") - 1 ))
new_cnt=$(tail -n +2 "$OUT/joined.tsv"  | awk -F'\t' '$5=="NEW"'     | wc -l)
cov_cnt=$(tail -n +2 "$OUT/joined.tsv"  | awk -F'\t' '$5=="COVERED"' | wc -l)
plan_cnt=$(tail -n +2 "$OUT/joined.tsv" | awk -F'\t' '$5=="IN-PLAN"' | wc -l)
gate_cnt=$(( $(wc -l < "$OUT/gates_issues.tsv") - 1 ))
stress_cnt=$(( $(wc -l < "$OUT/stress_leads.tsv") - 1 ))
prop_cnt=0
[[ -f "$OUT/proposed_tests.json" ]] && prop_cnt=$(grep -c '"status": "PROPOSE"' "$OUT/proposed_tests.json" 2>/dev/null || echo 0)
unmod_cnt=0
[[ -f "$OUT/proposed_tests.json" ]] && unmod_cnt=$(python3 -c "import json;print(json.load(open('$OUT/proposed_tests.json'))['summary']['unmodeled'])" 2>/dev/null || echo 0)
{
  echo "# extract-verbs-requirements -- summary"
  echo
  echo "## >>> START HERE: ${prop_cnt} test(s) to add"
  echo
  echo "See **PROPOSED_TESTS.md** (human) and **proposed_tests.json** (agent)."
  echo
  echo "- source: \`$SRC\`"
  echo "- coverage corpora: \`$(tail -n +1 "$CORPUS" 2>/dev/null | cut -f2 | sort -u | paste -sd', ' -)\`"
  echo "- plan cross-ref: \`${PLAN:-<none>}\`"
  echo "- distinct verbs/rdma API symbols: **$total** — COVERED: $cov_cnt, IN-PLAN: $plan_cnt, **NEW: $new_cnt**"
  echo "- gate/issue/probe signals: **$gate_cnt** · API-in-loop stress leads: **$stress_cnt**"
  echo
  echo "## Genuinely NEW symbols (not in any test corpus, not in the plan)"
  echo
  echo '| symbol | family | calls | sample location |'
  echo '|---|---|---|---|'
  tail -n +2 "$OUT/joined.tsv" | awk -F'\t' '$5=="NEW"{print "| `"$1"` | "$3" | "$4" | "$8" |"}' | head -60
  echo
  echo "## Known-issue / version-gate signals"
  echo
  echo '| kind | token | location |'
  echo '|---|---|---|'
  tail -n +2 "$OUT/gates_issues.tsv" | awk -F'\t' '{print "| "$1" | `"$2"` | "$3" |"}' | sort -u | head -80
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
