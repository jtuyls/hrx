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

iree_status_t to_native_buffer_type(
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_type_t* out_type) {
  switch (type) {
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY:
      *out_type = iree_hal_amdxdna_native_buffer_type_t::host_only;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE:
      *out_type = iree_hal_amdxdna_native_buffer_type_t::cacheable;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION:
      *out_type = iree_hal_amdxdna_native_buffer_type_t::instruction;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown amdxdna native buffer type");
  }
}

}  // namespace

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_c_type_t type,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  *out_buffer = nullptr;
  iree_hal_amdxdna_native_buffer_type_t native_type;
  IREE_RETURN_IF_ERROR(to_native_buffer_type(type, &native_type));
  iree_hal_amdxdna_native_buffer_ptr buffer;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_device_alloc_buffer(device, size, native_type,
                                                  &buffer));
  *out_buffer = buffer.release();
  return iree_ok_status();
}

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

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_buffer_sync_direction_t direction) {
  iree_hal_amdxdna_native_sync_direction_t native_direction;
  IREE_RETURN_IF_ERROR(to_native_sync_direction(direction, &native_direction));
  return iree_hal_amdxdna_native_buffer_sync_all(buffer, native_direction);
}

extern "C" iree_status_t iree_hal_amdxdna_native_buffer_c_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_ensure_allocated(buffer);
}

extern "C" uint64_t iree_hal_amdxdna_native_buffer_c_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_device_address(buffer);
}

extern "C" iree_device_size_t iree_hal_amdxdna_native_buffer_c_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_buffer_size(buffer);
}
