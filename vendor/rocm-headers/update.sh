#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Re-vendor the ROCm headers for a new release and prune to the minimal set the
# login-node build actually needs.
#
# Why this exists
# ---------------
# The suite builds on a login node with plain g++ and NO HIP toolchain: HIP/HSA
# entry points are resolved at run time via dlopen/dlsym, so the vendored headers
# are needed for DECLARATIONS ONLY. Upstream ships ~100 headers across three
# directories, but the host (declaration-only) compile pulls in only a couple of
# dozen. The device-side headers (fp16/bf16/cooperative_groups/texture device
# functions/...) are used solely by hipcc when regenerating the prebuilt kernel
# object (build_static_kernels.sh) against /opt/rocm -- never by this vendored
# set. This script therefore copies the three upstream directories, then keeps
# ONLY the transitive #include closure of the wrapper entry point HipApi.hpp.
#
# What it does
# ------------
#   1. Sparse-checkout the three upstream directories at the requested ref.
#   2. Stage them into a full temp tree and generate hip/hip_version.h.
#   3. Compute the exact header closure by running `g++ -M` over HipApi.hpp.
#   4. Replace vendor/rocm-headers/{hip,hsa} with ONLY the closure headers.
#   5. Refresh the README "Source" provenance (ref + resolved commit).
#   6. Verify the wrappers still syntax-check against the pruned set.
#
# Usage
# -----
#   vendor/rocm-headers/update.sh <upstream-ref>
#
#   # examples
#   vendor/rocm-headers/update.sh release/rocm-rel-7.15
#   vendor/rocm-headers/update.sh rocm-7.15.0
#
# Env overrides (all optional):
#   HIP_VERSION_MAJOR / HIP_VERSION_MINOR / HIP_VERSION_PATCH
#       Version macros written into hip_version.h. If unset, they are parsed from
#       the ref (e.g. ...7.15 -> 7.15.0); set them explicitly when the ref name
#       does not carry the version.
#   REPO   Upstream repo URL (default ROCm/rocm-systems).
#   KEEP_FULL=1  Skip pruning; vendor the full upstream copy (debug/escape hatch).
#   CXX    Compiler used for the closure/verify step (default g++).
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"          # vendor/rocm-headers
ROOT="$(cd "${HERE}/../.." && pwd)"            # hip-verbs-acceptance (project root)
HIPAPI="hip/utils/HipApi.hpp"                  # wrapper entry, relative to ROOT
CXX="${CXX:-g++}"
REPO="${REPO:-https://github.com/ROCm/rocm-systems.git}"

# Upstream directories we vendor and where they land under vendor/rocm-headers.
#   <upstream subdir>::<vendored subdir>
MAPPINGS=(
  "projects/hip/include/hip::hip"
  "projects/clr/hipamd/include/hip/amd_detail::hip/amd_detail"
  "projects/rocr-runtime/runtime/hsa-runtime/inc::hsa"
)

die() { echo "error: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: $(basename "$0") <upstream-ref>  (e.g. release/rocm-rel-7.15)"
REF="$1"

command -v "${CXX}" >/dev/null 2>&1 || die "compiler '${CXX}' not found"
command -v git >/dev/null 2>&1     || die "git not found"
[ -f "${ROOT}/${HIPAPI}" ]         || die "cannot find ${HIPAPI} under ${ROOT}"

# ---------------------------------------------------------------------------
# Version macros for hip_version.h (parse from ref unless overridden)
# ---------------------------------------------------------------------------
ver_from_ref="$(printf '%s\n' "${REF}" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
VMAJ="${HIP_VERSION_MAJOR:-${ver_from_ref%%.*}}"
_rest="${ver_from_ref#*.}"
VMIN="${HIP_VERSION_MINOR:-${_rest%%.*}}"
_patch="${ver_from_ref##*.}"
[ "${_patch}" = "${ver_from_ref}" ] && _patch=0   # ref had only MAJOR.MINOR
VPAT="${HIP_VERSION_PATCH:-${_patch}}"
[ -n "${VMAJ}" ] && [ -n "${VMIN}" ] && [ -n "${VPAT}" ] \
  || die "could not derive version from ref '${REF}'; set HIP_VERSION_MAJOR/MINOR/PATCH"

echo ">> re-vendoring ROCm headers"
echo "   ref:      ${REF}"
echo "   version:  ${VMAJ}.${VMIN}.${VPAT}"
echo "   repo:     ${REPO}"

# ---------------------------------------------------------------------------
# 1. Sparse-checkout the three upstream directories
# ---------------------------------------------------------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
CLONE="${WORK}/rocm-systems"

echo ">> cloning (sparse) ..."
git clone --quiet --filter=blob:none --no-checkout "${REPO}" "${CLONE}"
git -C "${CLONE}" checkout --quiet "${REF}"
git -C "${CLONE}" sparse-checkout init --cone
git -C "${CLONE}" sparse-checkout set \
  projects/hip/include \
  projects/clr/hipamd/include \
  projects/rocr-runtime/runtime/hsa-runtime/inc
git -C "${CLONE}" checkout --quiet
RESOLVED_COMMIT="$(git -C "${CLONE}" rev-parse HEAD)"
echo "   commit:   ${RESOLVED_COMMIT}"

# ---------------------------------------------------------------------------
# 2. Stage the FULL header tree (closure is computed against this)
# ---------------------------------------------------------------------------
FULL="${WORK}/full"
for m in "${MAPPINGS[@]}"; do
  src="${CLONE}/${m%%::*}"
  dst="${FULL}/${m##*::}"
  [ -d "${src}" ] || die "upstream dir missing at this ref: ${m%%::*}"
  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
done

# hip_version.h is generated by the CLR build (not in the source tree).
cat > "${FULL}/hip/hip_version.h" <<EOF
/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#ifndef HIP_VERSION_H
#define HIP_VERSION_H

// This header is normally generated by the ROCm/CLR build system. The HIP
// public headers (projects/hip) include it unconditionally, so the vendored
// copy provides a fixed definition. The actual HIP runtime version is resolved
// dynamically at run time via the dlopen wrapper; these macros only satisfy the
// vendored headers and gate optional declarations.
#define HIP_VERSION_MAJOR ${VMAJ}
#define HIP_VERSION_MINOR ${VMIN}
#define HIP_VERSION_PATCH ${VPAT}
#define HIP_VERSION_GITHASH ""
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME ""
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)

#endif // HIP_VERSION_H
EOF

# ---------------------------------------------------------------------------
# 3. Compute the transitive #include closure of the wrapper entry point.
#    g++ -M lists exactly the headers the host (declaration-only) build opens.
# ---------------------------------------------------------------------------
echo ">> computing header closure via ${CXX} -M ${HIPAPI} ..."
DEPS="${WORK}/deps.mk"
if ! ( cd "${ROOT}" && "${CXX}" -std=c++17 -fpermissive \
        -D__HIP_PLATFORM_AMD__ -DACCEPT_HAVE_HIP \
        -I"${FULL}" -I. \
        -M -include "${HIPAPI}" -xc++ /dev/null ) > "${DEPS}" 2> "${WORK}/deps.err"; then
  cat "${WORK}/deps.err" >&2
  die "closure computation failed (does the new release change an API the wrapper names?)"
fi

# Extract header paths under the staged full tree, as paths relative to it.
mapfile -t CLOSURE < <(tr ' \\' '\n\n' < "${DEPS}" \
  | grep -F "${FULL}/" \
  | sed "s#^${FULL}/##" \
  | sort -u)
[ "${#CLOSURE[@]}" -gt 0 ] || die "empty closure -- unexpected"
echo "   closure:  ${#CLOSURE[@]} headers"

# ---------------------------------------------------------------------------
# 4. Replace vendored hip/ and hsa/ with the closure (or the full tree).
# ---------------------------------------------------------------------------
echo ">> replacing vendored headers ..."
rm -rf "${HERE}/hip" "${HERE}/hsa"
if [ "${KEEP_FULL:-0}" = "1" ]; then
  cp -a "${FULL}/hip" "${HERE}/hip"
  cp -a "${FULL}/hsa" "${HERE}/hsa"
  echo "   KEEP_FULL=1: vendored full upstream copy ($(find "${HERE}"/hip "${HERE}"/hsa -type f | wc -l) files)"
else
  for h in "${CLOSURE[@]}"; do
    mkdir -p "${HERE}/$(dirname "${h}")"
    cp -a "${FULL}/${h}" "${HERE}/${h}"
  done
  echo "   pruned to ${#CLOSURE[@]} headers"
fi

# ---------------------------------------------------------------------------
# 5. Refresh README provenance (Branch + Commit lines under "## Source").
# ---------------------------------------------------------------------------
README="${HERE}/README.md"
if [ -f "${README}" ]; then
  sed -i \
    -e "s#^- Branch: .*#- Branch: \`${REF}\`#" \
    -e "s#^- Commit: .*#- Commit: \`${RESOLVED_COMMIT}\`#" \
    "${README}"
  echo "   README provenance updated"
fi

# ---------------------------------------------------------------------------
# 6. Verify the wrappers still syntax-check against the new vendored set.
# ---------------------------------------------------------------------------
echo ">> verifying wrappers compile against the new headers ..."
rc=0
( cd "${ROOT}" && "${CXX}" -std=c++17 -fpermissive -fsyntax-only \
    -DACCEPT_HAVE_HIP -I vendor/rocm-headers -I. \
    -include "${HIPAPI}" -xc++ /dev/null ) || rc=$?
[ "${rc}" -eq 0 ] || die "verification failed: wrappers do not compile against the new headers"

echo ">> done. Review 'git status' / 'git diff' under vendor/rocm-headers, then"
echo "   rebuild with -DACCEPT_WITH_HIP=ON and run the gate before committing."
