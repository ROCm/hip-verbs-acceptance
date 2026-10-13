#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# scan_api_surface.sh -- enumerate the HIP / CUDA-driver / HSA API surface a
# source tree actually CALLS, with file:line provenance.
#
# Distinguishes API *calls* (symbol immediately followed by '(') from mere
# identifiers (cudaDev, cudaArch are variables, not APIs). Counts call sites per
# symbol and records the first few locations.
#
# Usage:
#   scan_api_surface.sh <source-root> [out.tsv]
# Example:
#   scan_api_surface.sh /path/rocm-systems/projects/rccl/src rccl_api.tsv
#
# Output columns (TSV): symbol <TAB> family <TAB> call_count <TAB> sample_locations
# Read-only: never builds, never needs a device.

set -uo pipefail

ROOT="${1:?usage: scan_api_surface.sh <source-root> [out.tsv]}"
OUT="${2:-/dev/stdout}"

if [[ ! -d "$ROOT" ]]; then
  echo "scan_api_surface: source root not found: $ROOT" >&2
  exit 2
fi

# Symbol shape: hip<Upper>..., cu<Upper>..., cuda<Upper>..., hsa_amd_<lower>...
# Require a trailing '(' so we capture call sites, not type/var names.
PAT='\b(hip[A-Z][A-Za-z0-9_]+|cuda[A-Z][A-Za-z0-9_]+|cu[A-Z][A-Za-z0-9_]+|hsa_amd_[a-z0-9_]+)\('

family_of() {
  case "$1" in
    hsa_amd_*) echo "hsa" ;;
    cuda*)     echo "cuda-runtime" ;;
    cu[A-Z]*)  echo "cuda-driver" ;;
    hip*)      echo "hip" ;;
    *)         echo "other" ;;
  esac
}

# Collect every distinct called symbol.
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
    # All call sites of this exact symbol (symbol followed by '(').
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

echo "scan_api_surface: ${#SYMS[@]} distinct API symbols -> $OUT" >&2
