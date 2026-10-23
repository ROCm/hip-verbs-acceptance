/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#include "wrapper/hip/hip_api_wrapper.hpp"
#include "wrapper/hip/api_wrapper.hpp"

namespace acceptance {
namespace hipw {

lib_info_t hip_lib_info;
hip_lib_ops_t hip_lib_ops;

bool hip_api_init() {
  hip_lib_info.ops = &hip_lib_ops;
  hip_lib_info.fn_names = hip_fn_names;
  hip_lib_info.path = "libamdhip64.so";
  const int error = load_library(hip_lib_info);
  init_result().hip_lib_loaded = (error == HIPW_LOAD_LB_SUCCESS);
  return init_result().hip_lib_loaded;
}

void hip_api_fini() {
  close_library(hip_lib_info);
}

void* hip_dlopen_handle() {
  return hip_lib_info.handle;
}

}  // namespace hipw
}  // namespace acceptance
