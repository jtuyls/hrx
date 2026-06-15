// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_NATIVE_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_NATIVE_DEVICE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/native_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdxdna_native_c_command_opcode_t {
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU = 1,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF = 2,
  IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN = 3,
} iree_hal_amdxdna_native_c_command_opcode_t;

typedef enum iree_hal_amdxdna_native_c_power_mode_t {
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_LOW = 1,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_MEDIUM = 2,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_HIGH = 3,
  IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_TURBO = 4,
} iree_hal_amdxdna_native_c_power_mode_t;

typedef enum iree_hal_amdxdna_native_c_buffer_sync_model_t {
  IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_CALLER_SYNCS_BINDINGS = 0,
  IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_SUBMIT_SYNCS_BINDINGS = 1,
} iree_hal_amdxdna_native_c_buffer_sync_model_t;

enum iree_hal_amdxdna_native_c_context_image_model_bits_t {
  IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_PDI = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_XCLBIN = 1u << 1,
};

enum iree_hal_amdxdna_native_c_dispatch_model_bits_t {
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_CU = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU = 1u << 1,
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF = 1u << 2,
  IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN = 1u << 3,
};

enum iree_hal_amdxdna_native_c_completion_model_bits_t {
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_SYNCHRONOUS_WAIT = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_NATIVE_FENCE = 1u << 1,
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_PROGRESS_FENCE = 1u << 2,
  IREE_HAL_AMDXDNA_NATIVE_C_COMPLETION_MODEL_COMPLETION_SLOT = 1u << 3,
};

typedef struct iree_hal_amdxdna_native_c_device_caps_t {
  uint32_t ddi_version;
  uint32_t max_effective_queues;
  uint32_t max_command_chain_slots;
  uint32_t context_image_models;
  uint32_t dispatch_models;
  iree_hal_amdxdna_native_c_buffer_sync_model_t buffer_sync_model;
  uint32_t completion_models;
  bool supports_command_chain;
  bool supports_submit_many;
  bool supports_async_submit;
  bool supports_external_buffer_import;
  bool supports_external_buffer_export;
  bool supports_real_multi_queue;
  iree_hal_amdxdna_native_c_command_opcode_t default_dispatch_opcode;
} iree_hal_amdxdna_native_c_device_caps_t;

iree_status_t iree_hal_amdxdna_native_device_c_resolve_options(
    const struct iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    struct iree_hal_amdxdna_device_params* out_options,
    iree_byte_span_t* out_device_path_storage,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode);

iree_status_t iree_hal_amdxdna_native_device_c_create(
    const struct iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device);

void iree_hal_amdxdna_native_device_c_destroy(
    iree_hal_amdxdna_native_device_t* device);

iree_status_t iree_hal_amdxdna_native_device_c_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_power_mode_t power_mode);

iree_status_t iree_hal_amdxdna_native_device_c_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_NATIVE_DEVICE_H_
