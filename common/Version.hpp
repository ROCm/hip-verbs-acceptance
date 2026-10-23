/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// The harness's semantic version string (bumped by hand on release). The
// build-time git SHA is a SEPARATE, CMake-generated constant: see
// generated/common/version_info.hpp (cmake/GenerateVersion.cmake), which
// declares `acceptance::kGitSha` in this same namespace and is re-resolved on
// every `cmake --build` (not just at configure time).

namespace acceptance {

inline constexpr const char* kHarnessVersion = "0.1.0-phase1";

}  // namespace acceptance
