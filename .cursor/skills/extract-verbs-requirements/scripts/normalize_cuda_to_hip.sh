#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# normalize (verbs): no CUDA<->HIP mapping needed. Verbs symbols are already
# canonical (ibv_*/rdma_*/mlx5dv_*). Passthrough: echo the symbol unchanged.
# Kept for pipeline compatibility with the shared propose_tests/extract flow.
set -uo pipefail
[[ "${1:-}" == "--dump" ]] && { echo "verbs: identity normalization (no CUDA aliases)"; exit 0; }
echo "${1:-}"
