#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# scan_api_surface.sh -- enumerate the libibverbs / rdma-cm / mlx5dv API surface a
# source tree actually CALLS, with file:line provenance.
#
# Verbs counterpart of the HIP skill's scanner. Symbols are lowercase with a
# library prefix: ibv_*, rdma_*, mlx5dv_*. Distinguishes calls (symbol + '(')
# from type/struct names. Counts call sites per symbol.
#
# Usage:  scan_api_surface.sh <source-root> [out.tsv]
# Output (TSV): symbol <TAB> family <TAB> call_count <TAB> sample_locations
# Read-only.

set -uo pipefail
ROOT="${1:?usage: scan_api_surface.sh <source-root> [out.tsv]}"
OUT="${2:-/dev/stdout}"
[[ -d "$ROOT" ]] || { echo "scan_api_surface: source root not found: $ROOT" >&2; exit 2; }

# ibv_<lower>, rdma_<lower>, mlx5dv_<lower>, followed by '('
PAT='\b(ibv_[a-z0-9_]+|rdma_[a-z0-9_]+|mlx5dv_[a-z0-9_]+)\('

family_of() {
  case "$1" in
    ibv_*)    echo "ibverbs" ;;
    rdma_*)   echo "rdma-cm" ;;
    mlx5dv_*) echo "mlx5dv" ;;
    *)        echo "other" ;;
  esac
}

mapfile -t SYMS < <(
  grep -rhoE "$PAT" \
    --include='*.cc' --include='*.cpp' --include='*.c' \
    --include='*.h' --include='*.hpp' --include='*.cu' --include='*.cuh' \
    "$ROOT" 2>/dev/null | sed 's/($//; s/(//' | sort -u
)

{
  printf 'symbol\tfamily\tcall_count\tsample_locations\n'
  for sym in "${SYMS[@]}"; do
    [[ -z "$sym" ]] && continue
    locs=$(grep -rnE "\b${sym}\(" \
             --include='*.cc' --include='*.cpp' --include='*.c' \
             --include='*.h' --include='*.hpp' --include='*.cu' --include='*.cuh' \
             "$ROOT" 2>/dev/null)
    count=$(printf '%s\n' "$locs" | grep -c . )
    [[ "$count" -eq 0 ]] && continue
    sample=$(printf '%s\n' "$locs" | head -3 \
               | sed "s#^${ROOT%/}/##" | cut -d: -f1,2 | paste -sd';' -)
    printf '%s\t%s\t%s\t%s\n' "$sym" "$(family_of "$sym")" "$count" "$sample"
  done
} > "$OUT"

echo "scan_api_surface: ${#SYMS[@]} distinct verbs/rdma API symbols -> $OUT" >&2
