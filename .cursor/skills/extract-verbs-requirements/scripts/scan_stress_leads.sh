#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# scan_stress_leads.sh -- find places where RCCL calls a HIP/CUDA/HSA API INSIDE
# A LOOP. Those are the natural stress/scale acceptance scenarios (repeated
# alloc/map/register/launch), e.g. the 16x VMM-map ROCM-2550 repro. The plan
# should have a stress variant wherever the production code stresses an API.
#
# Heuristic: an API call whose line, or a line just above it within the same
# brace block, is a for/while/do loop header. Reports the loop site + the API.
#
# Usage:  scan_stress_leads.sh <source-root> [out.tsv] [window]
#   window (default 6): how many lines above the call to look for a loop header.
# Output (TSV): location <TAB> api <TAB> loop_line <TAB> loop_text
# Read-only; a lead generator, not proof of a hot loop.

set -uo pipefail
ROOT="${1:?usage: scan_stress_leads.sh <source-root> [out.tsv] [window]}"
OUT="${2:-/dev/stdout}"
WIN="${3:-6}"
[[ -d "$ROOT" ]] || { echo "scan_stress_leads: not found: $ROOT" >&2; exit 2; }

API_RE="${HIPREQ_API_RE:-(ibv_[a-z_]+|rdma_[a-z_]+|mlx5dv_[a-z_]+)}"

mapfile -t FILES < <(
  find "$ROOT" \( -name '*.cc' -o -name '*.cpp' -o -name '*.c' \
                  -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) 2>/dev/null
)

{
  printf 'location\tapi\tloop_line\tloop_text\n'
  for f in "${FILES[@]}"; do
    grep -qE "${API_RE}\(" "$f" 2>/dev/null || continue
    rel="${f#${ROOT%/}/}"
    awk -v API="$API_RE" -v WIN="$WIN" -v REL="$rel" '
      # Real iteration headers only: for(...) / while(...). Exclude do{}while
      # (almost always a CUDACHECK-style error macro, not a stress loop) and
      # backslash-continued macro lines.
      /(^|[^A-Za-z_])(for|while)[[:space:]]*\(/ && $0 !~ /\\[[:space:]]*$/ && $0 !~ /while[[:space:]]*\([[:space:]]*(0|false)[[:space:]]*\)/ {
        loopline=NR; looptext=$0
      }
      {
        if ($0 ~ /\\[[:space:]]*$/) next   # skip macro-body lines entirely
        callpat = API "[(]"
        if (match($0, callpat) && loopline>0 && NR-loopline<=WIN) {
          tok=substr($0,RSTART,RLENGTH); sym=substr(tok,1,length(tok)-1)
          lt=looptext; gsub(/^[[:space:]]+/,"",lt); lt=substr(lt,1,100)
          print REL":"NR"\t"sym"\t"loopline"\t"lt
        }
      }
    ' "$f"
  done
} > "$OUT"

n=$(( $(wc -l < "$OUT") - 1 ))
echo "scan_stress_leads: $n API-in-loop sites -> $OUT" >&2
