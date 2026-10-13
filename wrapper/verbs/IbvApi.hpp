/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once
// Single include for the verbs (libibverbs) dynamic wrapper.
//
//   wrapper/DynLib.hpp|.cpp                    — generic load_library (acceptance::wrap)
//   wrapper/verbs/ibv_api_wrapper.hpp|.cpp     — ibverbs ops table + redirects
//
// Call acceptance::ibvw::ibv_api_init() once at start-up (see main.cc).

#include "wrapper/verbs/ibv_api_wrapper.hpp"
