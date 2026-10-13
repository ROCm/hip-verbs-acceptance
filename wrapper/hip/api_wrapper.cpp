/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#include "wrapper/hip/api_wrapper.hpp"

#include "wrapper/hip/hip_api_wrapper.hpp"
#include "wrapper/hip/hsa_api_wrapper.hpp"
#include "wrapper/hip/hiprtc_api_wrapper.hpp"

namespace acceptance {
namespace hipw {

InitResult& init_result() {
  static InitResult r;
  return r;
}

void api_wrappers_init() {
  hip_api_init();
  hsa_api_init();
  hiprtc_api_init();
}

void api_wrappers_fini() {
  hip_api_fini();
  hsa_api_fini();
  hiprtc_api_fini();
}

const InitResult& init() {
  // Function-local static init is guaranteed thread-safe and run-once by the
  // C++11 standard (§6.7 [stmt.dcl]), so concurrent callers cannot
  // double-initialize the ops tables. The comma operator runs
  // api_wrappers_init() for its side effects, then yields init_result().
  static const InitResult& r = (api_wrappers_init(), init_result());
  return r;
}

}  // namespace hipw
}  // namespace acceptance
