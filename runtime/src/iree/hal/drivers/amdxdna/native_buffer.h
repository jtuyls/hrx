// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_NATIVE_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDXDNA_NATIVE_BUFFER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdxdna_native_buffer_t
    iree_hal_amdxdna_native_buffer_t;

typedef enum iree_hal_amdxdna_native_buffer_sync_direction_t {
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE = 0,
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST = 1,
} iree_hal_amdxdna_native_buffer_sync_direction_t;

typedef struct iree_hal_amdxdna_native_device_t
    iree_hal_amdxdna_native_device_t;

typedef enum iree_hal_amdxdna_native_buffer_c_type_t {
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY = 0,
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE = 1,
  IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION = 2,
} iree_hal_amdxdna_native_buffer_c_type_t;

iree_status_t iree_hal_amdxdna_native_device_c_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer);

void iree_hal_amdxdna_native_buffer_c_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_buffer_c_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr);

iree_status_t iree_hal_amdxdna_native_buffer_c_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset);

iree_status_t iree_hal_amdxdna_native_buffer_c_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction);

iree_status_t iree_hal_amdxdna_native_buffer_c_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer);

uint64_t iree_hal_amdxdna_native_buffer_c_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_device_size_t iree_hal_amdxdna_native_buffer_c_size(
    iree_hal_amdxdna_native_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_NATIVE_BUFFER_H_
