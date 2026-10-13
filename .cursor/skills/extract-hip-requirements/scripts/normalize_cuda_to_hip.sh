#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# normalize_cuda_to_hip.sh -- map a CUDA driver/runtime symbol to its HIP
# equivalent, so requirements grepped from RCCL/NCCL (which keep CUDA names)
# collapse onto the same canonical HIP symbol the tests and hardware use.
#
# Canonical layer = HIP name (that is what the standalone binary calls and what
# projects/hip-tests is indexed by). CUDA names are normalized on input.
#
# Source of truth, in priority order:
#   1) The repo's own hipify mapping, if reachable (HIPIFY CUDA2HIP tables).
#   2) cudawrap.h driver-PFN declarations (confirms which cu* RCCL wraps).
#   3) A small built-in fallback table for the common runtime/driver calls.
#
# Usage:
#   normalize_cuda_to_hip.sh <symbol> [hipify_map_file]
#   normalize_cuda_to_hip.sh --dump            # print the built-in table
# Prints the HIP symbol (or the input unchanged if already HIP / unknown).

set -uo pipefail

# --- built-in fallback: CUDA(driver|runtime) -> HIP --------------------------
# Mechanical rules cover most cases; exceptions are listed explicitly.
builtin_map() {
  case "$1" in
    # driver cuMem* VMM family -> hipMem*
    cuMemCreate)                       echo hipMemCreate ;;
    cuMemAddressReserve)               echo hipMemAddressReserve ;;
    cuMemAddressFree)                  echo hipMemAddressFree ;;
    cuMemMap)                          echo hipMemMap ;;
    cuMemUnmap)                        echo hipMemUnmap ;;
    cuMemRelease)                      echo hipMemRelease ;;
    cuMemSetAccess)                    echo hipMemSetAccess ;;
    cuMemGetAccess)                    echo hipMemGetAccess ;;
    cuMemGetAllocationGranularity)     echo hipMemGetAllocationGranularity ;;
    cuMemExportToShareableHandle)      echo hipMemExportToShareableHandle ;;
    cuMemImportFromShareableHandle)    echo hipMemImportFromShareableHandle ;;
    cuMemRetainAllocationHandle)       echo hipMemRetainAllocationHandle ;;
    cuMemGetAddressRange)              echo hipMemGetAddressRange ;;
    cuMemGetHandleForAddressRange)     echo hipMemGetHandleForAddressRange ;;
    cuPointerGetAttribute)             echo hipPointerGetAttribute ;;
    cuLaunchKernel)                    echo hipModuleLaunchKernel ;;
    cuLaunchKernelEx)                  echo hipLaunchKernelEx ;;
    cuDeviceGetAttribute)             echo hipDeviceGetAttribute ;;
    cuGetErrorString)                  echo hipDrvGetErrorString ;;
    cuStreamBatchMemOp)                echo hipStreamBatchMemOp ;;
    # multicast (no stable HIP equivalent yet -> keep, flag)
    cuMulticast*)                      echo "${1}:NO_HIP_EQUIV" ;;
    # runtime cuda* -> hip* : mechanical s/^cuda/hip/
    cuda*)                             echo "hip${1#cuda}" ;;
    # already HIP or HSA: unchanged
    hip*|hsa_amd_*)                    echo "$1" ;;
    # unknown driver symbol: keep, flag for manual mapping
    cu[A-Z]*)                          echo "${1}:CHECK_MAP" ;;
    *)                                 echo "$1" ;;
  esac
}

if [[ "${1:-}" == "--dump" ]]; then
  for s in cuMemCreate cuMemMap cuLaunchKernel cuLaunchKernelEx cuDeviceGetAttribute \
           cudaStreamCreateWithFlags cudaEventCreateWithFlags cudaMemcpyAsync \
           cuMulticastCreate cuPointerGetAttribute; do
    printf '%s\t->\t%s\n' "$s" "$(builtin_map "$s")"
  done
  exit 0
fi

SYM="${1:?usage: normalize_cuda_to_hip.sh <symbol> [hipify_map_file]}"
MAP="${2:-}"

# 1) repo hipify table, if provided and it has the symbol
if [[ -n "$MAP" && -f "$MAP" ]]; then
  hit=$(grep -E "\"$SYM\"" "$MAP" 2>/dev/null | grep -oE '"hip[A-Za-z0-9_]+"' | head -1 | tr -d '"')
  if [[ -n "$hit" ]]; then echo "$hit"; exit 0; fi
fi

# 2/3) fallback rules
builtin_map "$SYM"
