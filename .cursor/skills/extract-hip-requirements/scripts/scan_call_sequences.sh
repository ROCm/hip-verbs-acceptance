#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# scan_call_sequences.sh -- emit HIP/CUDA/HSA API calls IN SOURCE ORDER, grouped
# by file and best-effort function region, so a reader can reconstruct the
# *sequences* of primitives (e.g. MemCreate -> AddressReserve -> Map -> SetAccess)
# that are the actual acceptance scenarios -- not just the dedup'd symbol set.
#
# Why this matters: scan_api_surface.sh answers "which APIs are used"; this answers
# "in what order, together" -- the difference between a symbol and a scenario.
#
# C++ function boundaries are not reliably regex-parseable (templates, multi-line
# signatures, namespaces). So this groups by a best-effort region marker: the
# nearest preceding line that looks like a function/method header. The model
# refines the exact sequence by reading the cited file:line spans; this is a lead
# generator, not an AST.
#
# Usage:  scan_call_sequences.sh <source-root> [out.txt] [min_distinct]
#   min_distinct (default 3): only print regions with >= N distinct APIs.
# Output: human-readable blocks:  <file>  @<header-line> <header>
#                                   <line>  <api>
# Read-only.

set -uo pipefail
ROOT="${1:?usage: scan_call_sequences.sh <source-root> [out.txt] [min_distinct]}"
OUT="${2:-/dev/stdout}"
MIN="${3:-3}"
[[ -d "$ROOT" ]] || { echo "scan_call_sequences: not found: $ROOT" >&2; exit 2; }

# API symbol pattern WITHOUT the trailing '(' — the awk body appends it safely.
API_RE='(cu[A-Z]|cuda[A-Z]|hip[A-Z]|hsa_amd_)[A-Za-z0-9_]+'
# "looks like a function header": starts at column 0 with an identifier and has '('.
HDR_RE='^[A-Za-z_].*[A-Za-z0-9_]+[[:space:]]*[(]'

mapfile -t FILES < <(
  find "$ROOT" \( -name '*.cc' -o -name '*.cpp' -o -name '*.c' \
                  -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) 2>/dev/null
)

{
  for f in "${FILES[@]}"; do
    # Does the file call any API at all? skip fast if not.
    grep -qE "${API_RE}\(" "$f" 2>/dev/null || continue
    rel="${f#${ROOT%/}/}"
    awk -v API="$API_RE" -v HDR="$HDR_RE" -v MIN="$MIN" -v REL="$rel" '
      function flush() {
        if (ndist >= MIN) {
          printf "\n%s  @%d %s\n", REL, hline, hdr
          for (i=1;i<=ncall;i++) printf "  %s\n", calls[i]
        }
        ncall=0; ndist=0; delete seen
      }
      # header candidate: not a control keyword, not ending in ";" (decl), has "("
      $0 ~ HDR && $0 !~ /^[[:space:]]*(if|for|while|switch|return|else)\b/ && $0 !~ /;[[:space:]]*$/ {
        # only treat as new region if it is plausibly a definition (has { now or soon)
        flush(); hdr=$0; gsub(/[[:space:]]+/," ",hdr); hline=NR
      }
      {
        line=$0
        callpat = API "[(]"
        while (match(line, callpat)) {
          tok=substr(line, RSTART, RLENGTH)
          sym=substr(tok, 1, length(tok)-1)
          ncall++; calls[ncall]=NR"  "sym
          if (!(sym in seen)) { seen[sym]=1; ndist++ }
          line=substr(line, RSTART+RLENGTH)
        }
      }
      END { flush() }
    ' "$f"
  done
} > "$OUT"

n=$(grep -c '^[^ ]' "$OUT" 2>/dev/null || echo 0)
echo "scan_call_sequences: $n candidate regions (>= ${MIN} distinct APIs) -> $OUT" >&2
