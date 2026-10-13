/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#include "wrapper/hip/hsa_api_wrapper.hpp"
#include "wrapper/hip/api_wrapper.hpp"

namespace acceptance {
namespace hipw {

lib_info_t hsa_lib_info;
hsa_lib_ops_t hsa_lib_ops;

bool hsa_api_init() {
  hsa_lib_info.ops = &hsa_lib_ops;
  hsa_lib_info.fn_names = hsa_fn_names;
  hsa_lib_info.path = "libhsa-runtime64.so";
  const int error = load_library(hsa_lib_info);
  init_result().hsa_lib_loaded = (error == HIPW_LOAD_LB_SUCCESS);
  return init_result().hsa_lib_loaded;
}

void hsa_api_fini() {
  close_library(hsa_lib_info);
}

}  // namespace hipw
}  // namespace acceptance
