/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// Single include for the HIP/HSA dynamic wrapper (oneCCL layout).
//
//   wrapper/DynLib.hpp|.cpp                  — generic load_library() (acceptance::wrap)
//   wrapper/hip/api_wrapper.hpp|.cpp         — HIP InitResult + api_wrappers_init()
//   wrapper/hip/hip_api_wrapper.hpp|.cpp     — HIP ops table + redirects
//   wrapper/hip/hsa_api_wrapper.hpp|.cpp     — HSA ops table + redirects
//
// Call acceptance::hipw::init() once at start-up (see main.cc).

#include "wrapper/hip/api_wrapper.hpp"
#include "wrapper/hip/hip_api_wrapper.hpp"
#include "wrapper/hip/hsa_api_wrapper.hpp"
#include "wrapper/hip/hiprtc_api_wrapper.hpp"
