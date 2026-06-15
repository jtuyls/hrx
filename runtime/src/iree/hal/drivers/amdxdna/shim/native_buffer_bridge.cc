// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/native_buffer.h"

#include "iree/hal/drivers/amdxdna/native.h"

namespace {

iree_status_t to_native_sync_direction(
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_hal_amdxdna_native_sync_direction_t* out_direction) {
  switch (direction) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE:
      *out_direction =
          iree_hal_amdxdna_native_sync_direction_t::host_to_device;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST:
      *out_direction =
          iree_hal_amdxdna_native_sync_direction_t::device_to_host;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown amdxdna native buffer sync direction");
  }
}

}  // namespace

extern "C" void iree_hal_amdxdna_native_buffer_c_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  iree_hal_amdxdna_native_buffer_destroy(buffer);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  return iree_hal_amdxdna_native_buffer_map(buffer, out_ptr);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction,
    iree_device_size_t size, iree_device_size_t offset) {
  iree_hal_amdxdna_native_sync_direction_t native_direction;
  IREE_RETURN_IF_ERROR(to_native_sync_direction(direction, &native_direction));
  return iree_hal_amdxdna_native_buffer_sync(buffer, native_direction, size,
                                            offset);
}
