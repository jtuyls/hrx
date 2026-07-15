// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library.h"

static int elementwise_mul_dispatch(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  (void)workgroup_state;
  const float* lhs = (const float*)dispatch_state->binding_ptrs[0];
  const float* rhs = (const float*)dispatch_state->binding_ptrs[1];
  float* out = (float*)dispatch_state->binding_ptrs[2];
  for (int i = 0; i < 4; ++i) {
    out[i] = lhs[i] * rhs[i];
  }
  return 0;
}

static const iree_hal_executable_library_header_t header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "ex",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

static const iree_hal_executable_dispatch_v0_t entry_points[1] = {
    elementwise_mul_dispatch,
};

static const iree_hal_executable_dispatch_attrs_v0_t entry_attrs[1] = {
    {
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .binding_count = 3,
        .workgroup_size_x = 1,
        .workgroup_size_y = 1,
        .workgroup_size_z = 1,
        .parameter_count = 0,
        .constant_byte_length = 0,
    },
};

static const char* entry_point_names[1] = {
    "elementwise_mul",
};

static const iree_hal_executable_library_v0_t library = {
    .header = &header,
    .imports =
        {
            .count = 0,
            .symbols = 0,
        },
    .exports =
        {
            .count = 1,
            .ptrs = entry_points,
            .attrs = entry_attrs,
            .params = 0,
            .occupancy = 0,
            .names = entry_point_names,
            .tags = 0,
            .parameter_names = 0,
            .source_locations = 0,
            .stage_locations = 0,
        },
    .constants =
        {
            .count = 0,
        },
    .sources =
        {
            .count = 0,
            .files = 0,
        },
};

IREE_HAL_EXECUTABLE_LIBRARY_EXPORT const iree_hal_executable_library_header_t**
iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  (void)environment;
  return max_version >= IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST
             ? (const iree_hal_executable_library_header_t**)&library
             : 0;
}
