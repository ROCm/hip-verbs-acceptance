#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# build_coverage_corpus.sh -- index EVERYWHERE a HIP symbol is already tested in
# the monorepo, not just projects/hip-tests. Answers "what already exists for HIP"
# across the whole repo so the requirements delta distinguishes:
#   COVERED-UNIT (and in which test suite) vs IN-PLAN vs genuinely NEW.
#
# Builds one TSV: hip_symbol <TAB> source_tag <TAB> first_path
# where source_tag is the test corpus root that contains it (hip-tests, rccl-test,
# rocshmem, ...). A symbol can appear under several tags (multiple owners).
#
# Usage:
#   build_coverage_corpus.sh <out.tsv> [root1[:tag1]] [root2[:tag2]] ...
# If no roots are given, a built-in default set is discovered via resolve_source.sh
# search roots (hip-tests, rccl/test, rocshmem/tests, rccl-tests).
# Read-only.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: build_coverage_corpus.sh <out.tsv> [root[:tag]]...}"; shift || true

# --- assemble (root, tag) list ----------------------------------------------
declare -a ROOTS TAGS
add_root() { # path tag
  [[ -d "$1" ]] || return 0
  ROOTS+=("$1"); TAGS+=("$2")
}

if [[ $# -gt 0 ]]; then
  for spec in "$@"; do
    if [[ "$spec" == *:* ]]; then add_root "${spec%%:*}" "${spec##*:}"
    else add_root "$spec" "$(basename "$spec")"; fi
  done
else
  # Default monorepo test corpora. Resolve via the repo locator where possible.
  ht="$(bash "$HERE/resolve_source.sh" --hip-tests 2>/dev/null)"; add_root "$ht" "hip-tests"
  # rccl/test and others sit next to the resolved rccl src.
  rcclsrc="$(bash "$HERE/resolve_source.sh" rccl 2>/dev/null)"
  if [[ -n "$rcclsrc" ]]; then
    repo="$(dirname "$rcclsrc")"; proj="$(dirname "$repo")"
    add_root "$repo/test" "rccl-test"
    add_root "$proj/rccl-tests" "rccl-tests"
    add_root "$proj/rocshmem/tests" "rocshmem-tests"
  fi
fi

if [[ ${#ROOTS[@]} -eq 0 ]]; then
  echo "build_coverage_corpus: no test roots found" >&2
  : > "$OUT"; exit 0
fi

# --- index each root ---------------------------------------------------------
{
  for i in "${!ROOTS[@]}"; do
    root="${ROOTS[$i]}"; tag="${TAGS[$i]}"
    grep -rEno '\bhip[A-Z][A-Za-z0-9_]+' \
      --include='*.cc' --include='*.cpp' --include='*.h' --include='*.hpp' \
      --include='*.cu' --include='*.cuh' \
      "$root" 2>/dev/null \
      | sed "s#^${root%/}/##" \
      | awk -F: -v tag="$tag" '
          { sym=$3; path=$1; key=sym SUBSEP tag;
            if (!(key in seen)) { seen[key]=1; print sym"\t"tag"\t"path } }'
  done
} | sort -u > "$OUT"

ntags=${#ROOTS[@]}; nsyms=$(cut -f1 "$OUT" | sort -u | wc -l)
echo "build_coverage_corpus: indexed $ntags corpora, $nsyms distinct HIP symbols -> $OUT" >&2
for i in "${!ROOTS[@]}"; do echo "  [${TAGS[$i]}] ${ROOTS[$i]}" >&2; done
