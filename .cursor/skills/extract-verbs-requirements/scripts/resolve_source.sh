#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# resolve_source.sh -- turn a repo NAME or a path into a concrete source dir to
# scan, autonomously. Lets the skill work on any repo (rccl, nccl, rocshmem,
# nixl, mooncake, ...) without hardcoded paths.
#
# Resolution order for the argument:
#   1. If it is an existing directory -> use it directly (after src-subdir detect).
#   2. If it is a known alias (rccl/nccl/rocshmem/...) -> search SEARCH_ROOTS for it.
#   3. Otherwise treat it as a repo dir name and search SEARCH_ROOTS for it.
# Then pick the best SOURCE SUBDIR inside the repo (src/, library/src, catch/, ...).
#
# Search roots (override with env HIPREQ_SEARCH_ROOTS, ':'-separated):
#   $HIPREQ_SEARCH_ROOTS, then a built-in list of common monorepo locations.
#
# Usage:
#   resolve_source.sh <name|path>            # prints: <resolved-source-dir>
#   resolve_source.sh --hip-tests            # prints the hip-tests reference dir
#   resolve_source.sh --list                 # list discoverable repos
# Exit non-zero (and print nothing) if it cannot resolve.

set -uo pipefail

# --- search roots ------------------------------------------------------------
# Derived from WHERE THE SKILL IS RUN, not any hardcoded user path. Order:
#   1) $HIPREQ_SEARCH_ROOTS (explicit override, ':'-separated), if set
#   2) the current working directory and a few ancestors (so running inside a
#      checkout "just works")
#   3) if CWD is inside a git monorepo, its root + a conventional projects/ dir
git_root() { git -C "$PWD" rev-parse --show-toplevel 2>/dev/null; }
default_roots() {
  local r=()
  [[ -n "${HIPREQ_SEARCH_ROOTS:-}" ]] && IFS=':' read -ra r <<< "$HIPREQ_SEARCH_ROOTS"
  r+=( "$PWD" "$PWD/.." "$PWD/../.." )
  local gr; gr="$(git_root)"
  [[ -n "$gr" ]] && r+=( "$gr" "$gr/projects" )
  printf '%s\n' "${r[@]}"
}

# --- per-repo source-subdir preference --------------------------------------
# Given a repo root, echo the dir that actually holds the C/C++ sources to scan.
pick_src_subdir() {
  local repo="$1"
  for cand in src library/src catch source lib; do
    if [[ -d "$repo/$cand" ]] && _has_cxx "$repo/$cand"; then
      echo "$repo/$cand"; return 0
    fi
  done
  # fall back to the repo root if it directly contains sources
  if _has_cxx "$repo"; then echo "$repo"; return 0; fi
  return 1
}

_has_cxx() { # true if dir has any C/C++ source within 4 levels
  find "$1" -maxdepth 4 \( -name '*.cc' -o -name '*.cpp' -o -name '*.cu' \
       -o -name '*.h' -o -name '*.hpp' \) -print -quit 2>/dev/null | grep -q .
}

# --- alias -> candidate dir names -------------------------------------------
alias_dirs() {
  case "$1" in
    rccl)      echo "rccl" ;;
    nccl)      echo "nccl" ;;
    rocshmem)  echo "rocshmem rocSHMEM" ;;
    nixl)      echo "nixl NIXL" ;;
    mooncake)  echo "mooncake Mooncake" ;;
    hip-tests|hiptests) echo "hip-tests hip_tests" ;;
    *)         echo "$1" ;;   # treat as literal dir name
  esac
}

find_repo() { # <name> -> repo root path (first hit)
  local name="$1" root cand
  while IFS= read -r root; do
    [[ -d "$root" ]] || continue
    for cand in $(alias_dirs "$name"); do
      if [[ -d "$root/$cand" ]]; then echo "$root/$cand"; return 0; fi
    done
  done < <(default_roots)
  # deeper search as a last resort (bounded)
  while IFS= read -r root; do
    [[ -d "$root" ]] || continue
    for cand in $(alias_dirs "$name"); do
      local hit
      hit=$(find "$root" -maxdepth 3 -type d -name "$cand" -print -quit 2>/dev/null)
      [[ -n "$hit" ]] && { echo "$hit"; return 0; }
    done
  done < <(default_roots)
  return 1
}

# --- modes -------------------------------------------------------------------
if [[ "${1:-}" == "--hip-tests" ]]; then
  repo=$(find_repo hip-tests) || { echo "resolve_source: hip-tests not found" >&2; exit 1; }
  echo "$repo"; exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  while IFS= read -r root; do
    [[ -d "$root" ]] || continue
    for d in "$root"/*/; do
      [[ -d "$d" ]] || continue
      _has_cxx "$d" && printf '%s\t%s\n' "$(basename "$d")" "$d"
    done
  done < <(default_roots) | sort -u
  exit 0
fi

ARG="${1:?usage: resolve_source.sh <name|path|.> | --hip-tests | --list}"

# 0) "." or "cwd" or empty -> the current working directory (preferred default:
#    run the skill inside a checkout and it scans that checkout).
if [[ "$ARG" == "." || "$ARG" == "cwd" ]]; then
  pick_src_subdir "$PWD" || { echo "resolve_source: no C/C++ sources under CWD ($PWD)" >&2; exit 1; }
  exit 0
fi

# 1) explicit existing dir
if [[ -d "$ARG" ]]; then
  pick_src_subdir "$ARG" || { echo "resolve_source: no C/C++ sources under $ARG" >&2; exit 1; }
  exit 0
fi

# 2/3) name / alias -> repo -> src subdir (searched relative to CWD / git root)
repo=$(find_repo "$ARG") || true
if [[ -z "$repo" ]]; then
  # Fallback: if CWD itself is (or contains) the sources, just use it.
  if pick_src_subdir "$PWD" >/dev/null 2>&1; then
    pick_src_subdir "$PWD"; exit 0
  fi
  echo "resolve_source: could not locate '$ARG' near CWD; pass a path or run inside the repo" >&2
  exit 1
fi
pick_src_subdir "$repo" || { echo "resolve_source: located '$repo' but found no C/C++ sources" >&2; exit 1; }
