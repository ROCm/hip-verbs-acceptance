/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#include "wrapper/hip/hiprtc_api_wrapper.hpp"
#include "wrapper/hip/api_wrapper.hpp"

namespace acceptance {
namespace hipw {

lib_info_t hiprtc_lib_info;
hiprtc_lib_ops_t hiprtc_lib_ops;

bool hiprtc_api_init() {
  hiprtc_lib_info.ops = &hiprtc_lib_ops;
  hiprtc_lib_info.fn_names = hiprtc_fn_names;
  hiprtc_lib_info.path = "libhiprtc.so";
  const int error = load_library(hiprtc_lib_info);
  init_result().hiprtc_lib_loaded = (error == HIPW_LOAD_LB_SUCCESS);
  return init_result().hiprtc_lib_loaded;
}

void hiprtc_api_fini() {
  close_library(hiprtc_lib_info);
}

}  // namespace hipw
}  // namespace acceptance
