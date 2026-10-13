/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hiprtc.h>

#include <string>
#include <vector>

namespace acceptance {
namespace hipw {

// hiprtc (HIP runtime compilation) ops table. Kernels are kept as source text in
// the host binary and compiled for the *current* device at run time, so the suite
// needs no device compiler at build time and no GPU-arch is baked into anything.
// Loaded from libhiprtc.so via the same dlopen wrapper as HIP/HSA.
typedef struct hiprtc_lib_ops_t {
  decltype(::hiprtcCompileProgram)* hiprtcCompileProgram;
  decltype(::hiprtcCreateProgram)* hiprtcCreateProgram;
  decltype(::hiprtcDestroyProgram)* hiprtcDestroyProgram;
  decltype(::hiprtcGetCode)* hiprtcGetCode;
  decltype(::hiprtcGetCodeSize)* hiprtcGetCodeSize;
  decltype(::hiprtcGetErrorString)* hiprtcGetErrorString;
  decltype(::hiprtcGetProgramLog)* hiprtcGetProgramLog;
  decltype(::hiprtcGetProgramLogSize)* hiprtcGetProgramLogSize;
} hiprtc_lib_ops_t;

static std::vector<std::string> hiprtc_fn_names = {
    "hiprtcCompileProgram",
    "hiprtcCreateProgram",
    "hiprtcDestroyProgram",
    "hiprtcGetCode",
    "hiprtcGetCodeSize",
    "hiprtcGetErrorString",
    "hiprtcGetProgramLog",
    "hiprtcGetProgramLogSize",
};

extern hiprtc_lib_ops_t hiprtc_lib_ops;

#define hiprtcCompileProgram ::acceptance::hipw::hiprtc_lib_ops.hiprtcCompileProgram
#define hiprtcCreateProgram ::acceptance::hipw::hiprtc_lib_ops.hiprtcCreateProgram
#define hiprtcDestroyProgram ::acceptance::hipw::hiprtc_lib_ops.hiprtcDestroyProgram
#define hiprtcGetCode ::acceptance::hipw::hiprtc_lib_ops.hiprtcGetCode
#define hiprtcGetCodeSize ::acceptance::hipw::hiprtc_lib_ops.hiprtcGetCodeSize
#define hiprtcGetErrorString ::acceptance::hipw::hiprtc_lib_ops.hiprtcGetErrorString
#define hiprtcGetProgramLog ::acceptance::hipw::hiprtc_lib_ops.hiprtcGetProgramLog
#define hiprtcGetProgramLogSize ::acceptance::hipw::hiprtc_lib_ops.hiprtcGetProgramLogSize

bool hiprtc_api_init();

void hiprtc_api_fini();

}  // namespace hipw
}  // namespace acceptance
