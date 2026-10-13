#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Regenerate the vendored multi-arch static-kernel object (kernels_static.o).
#
# This is the ONLY build step that needs a HIP device compiler. Run it ONCE on a
# node that has hipcc (e.g. a build/GPU node) and commit the resulting
# prebuilt/kernels_static.o; the login-node build then just links that object
# with no HIP toolchain present.
#
# The object is a fat (multi-arch) relocatable: it carries device ISA for every
# arch in ARCHES, and the runtime selects the matching one at load. Add a new
# arch here and re-run when a new GPU generation appears.
#
# Usage:
#   [HIPCC=/opt/rocm/bin/hipcc] [ARCHES="gfx908 gfx90a gfx942 gfx950"] \
#     hip/kernel_launch/build_static_kernels.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HIPCC="${HIPCC:-hipcc}"
ARCHES="${ARCHES:-gfx908 gfx90a gfx942 gfx950}"

offload=()
for a in ${ARCHES}; do offload+=("--offload-arch=${a}"); done

mkdir -p "${HERE}/prebuilt"
out="${HERE}/prebuilt/kernels_static.o"

echo "Building ${out}"
echo "  hipcc:  ${HIPCC}"
echo "  arches: ${ARCHES}"
# -fPIC: the suite executable links as PIE, so the host side of this relocatable
# object must use position-independent relocations.
"${HIPCC}" -c -fPIC "${offload[@]}" "${HERE}/static_kernels.cpp" -o "${out}"
echo "done: ${out}"
