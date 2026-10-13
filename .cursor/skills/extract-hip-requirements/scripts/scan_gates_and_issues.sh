#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# scan_gates_and_issues.sh -- find version gates and known-issue references that
# turn a plain API call into a CONDITIONAL or REGRESSION acceptance requirement.
#
# Three signal classes:
#   GATE    -- compile-time / runtime version guards (HIP_VERSION, CUDART_VERSION,
#              CUDA_VERSION, ROCM_VERSION, NCCL_CUMEM_VERSION_SUPPORTED, and
#              numeric driver-version comparisons like >= 71200000).
#   ISSUE   -- JIRA / tracker IDs in comments (e.g. ABC-1234).
#   PROBE   -- runtime capability probes (ncclCe*Supported, *Supported(), can*()).
#
# Usage:  scan_gates_and_issues.sh <source-root> [out.tsv]
# Output (TSV): kind <TAB> token <TAB> location <TAB> line_text
# Read-only.

set -uo pipefail
ROOT="${1:?usage: scan_gates_and_issues.sh <source-root> [out.tsv]}"
OUT="${2:-/dev/stdout}"
[[ -d "$ROOT" ]] || { echo "scan_gates_and_issues: not found: $ROOT" >&2; exit 2; }

INC=( --include='*.cc' --include='*.cpp' --include='*.c'
      --include='*.h'  --include='*.hpp' --include='*.cu' --include='*.cuh' )

emit() { # kind regex
  local kind="$1" rx="$2"
  grep -rnE "$rx" "${INC[@]}" "$ROOT" 2>/dev/null \
    | while IFS= read -r line; do
        local loc text tok
        loc=$(printf '%s' "$line" | cut -d: -f1,2 | sed "s#^${ROOT%/}/##")
        text=$(printf '%s' "$line" | cut -d: -f3- | sed 's/^[[:space:]]*//' | cut -c1-160)
        tok=$(printf '%s' "$text" | grep -oE "$rx" | head -1)
        printf '%s\t%s\t%s\t%s\n' "$kind" "$tok" "$loc" "$text"
      done
}

{
  printf 'kind\ttoken\tlocation\tline_text\n'
  emit GATE  'HIP_VERSION[[:space:]]*[<>=!]+[[:space:]]*[0-9]+'
  emit GATE  '(CUDART_VERSION|CUDA_VERSION|ROCM_VERSION)[[:space:]]*[<>=!]+[[:space:]]*[0-9]+'
  emit GATE  'NCCL_CUMEM_VERSION_SUPPORTED'
  emit GATE  '(driverVersion|runtimeVersion)[[:space:]]*[<>=!]+[[:space:]]*[0-9]{6,}'
  emit ISSUE '[A-Z][A-Z0-9]+-[0-9]+'
  emit PROBE '[A-Za-z_][A-Za-z0-9_]*Supported[[:space:]]*\('
} > "$OUT"

echo "scan_gates_and_issues: gates/issues/probes -> $OUT" >&2
