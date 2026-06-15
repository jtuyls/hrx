// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/native_device.h"

#include <cstring>
#include <string>

#include "iree/hal/drivers/amdxdna/native.h"

namespace {

iree_hal_amdxdna_native_c_command_opcode_t to_c_command_opcode(
    iree_hal_amdxdna_native_command_opcode_t opcode) {
  switch (opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu:
      return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf:
      return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF;
    case iree_hal_amdxdna_native_command_opcode_t::command_chain:
      return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN;
  }
  return IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU;
}

iree_hal_amdxdna_native_power_mode_t from_c_power_mode(
    iree_hal_amdxdna_native_c_power_mode_t power_mode) {
  switch (power_mode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT:
      return iree_hal_amdxdna_native_power_mode_t::default_mode;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_LOW:
      return iree_hal_amdxdna_native_power_mode_t::low;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_MEDIUM:
      return iree_hal_amdxdna_native_power_mode_t::medium;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_HIGH:
      return iree_hal_amdxdna_native_power_mode_t::high;
    case IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_TURBO:
      return iree_hal_amdxdna_native_power_mode_t::turbo;
  }
  return iree_hal_amdxdna_native_power_mode_t::default_mode;
}

iree_hal_amdxdna_native_c_power_mode_t to_c_power_mode(
    iree_hal_amdxdna_native_power_mode_t power_mode) {
  switch (power_mode) {
    case iree_hal_amdxdna_native_power_mode_t::default_mode:
      return IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT;
    case iree_hal_amdxdna_native_power_mode_t::low:
      return IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_LOW;
    case iree_hal_amdxdna_native_power_mode_t::medium:
      return IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_MEDIUM;
    case iree_hal_amdxdna_native_power_mode_t::high:
      return IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_HIGH;
    case iree_hal_amdxdna_native_power_mode_t::turbo:
      return IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_TURBO;
  }
  return IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT;
}

iree_hal_amdxdna_native_c_buffer_sync_model_t to_c_buffer_sync_model(
    iree_hal_amdxdna_native_buffer_sync_model_t model) {
  switch (model) {
    case iree_hal_amdxdna_native_buffer_sync_model_t::caller_syncs_bindings:
      return IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_CALLER_SYNCS_BINDINGS;
    case iree_hal_amdxdna_native_buffer_sync_model_t::submit_syncs_bindings:
      return IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_SUBMIT_SYNCS_BINDINGS;
  }
  return IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_CALLER_SYNCS_BINDINGS;
}

iree_hal_amdxdna_native_c_device_caps_t to_c_device_caps(
    const iree_hal_amdxdna_native_device_caps_t& caps) {
  iree_hal_amdxdna_native_c_device_caps_t c_caps = {};
  c_caps.ddi_version = caps.ddi_version;
  c_caps.max_effective_queues = caps.max_effective_queues;
  c_caps.max_command_chain_slots = caps.max_command_chain_slots;
  c_caps.context_image_models = caps.context_image_models;
  c_caps.dispatch_models = caps.dispatch_models;
  c_caps.buffer_sync_model = to_c_buffer_sync_model(caps.buffer_sync_model);
  c_caps.completion_models = caps.completion_models;
  c_caps.supports_command_chain = caps.supports_command_chain;
  c_caps.supports_submit_many = caps.supports_submit_many;
  c_caps.supports_async_submit = caps.supports_async_submit;
  c_caps.supports_external_buffer_import =
      caps.supports_external_buffer_import;
  c_caps.supports_external_buffer_export =
      caps.supports_external_buffer_export;
  c_caps.supports_real_multi_queue = caps.supports_real_multi_queue;
  c_caps.default_dispatch_opcode =
      to_c_command_opcode(caps.default_dispatch_opcode);
  return c_caps;
}

}  // namespace

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_resolve_options(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_device_params* out_options,
    iree_byte_span_t* out_device_path_storage,
    iree_hal_amdxdna_native_c_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_device_path_storage = iree_byte_span_empty();
  iree_hal_amdxdna_native_power_mode_t native_power_mode =
      iree_hal_amdxdna_native_power_mode_t::default_mode;
  std::string device_path_storage;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_resolve_device_options(
      options, out_options, &device_path_storage, &native_power_mode,
      out_should_set_power_mode));
  if (!device_path_storage.empty()) {
    uint8_t* storage = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, device_path_storage.size(), (void**)&storage));
    std::memcpy(storage, device_path_storage.data(),
                device_path_storage.size());
    *out_device_path_storage =
        iree_make_byte_span(storage, device_path_storage.size());
    out_options->device_path =
        iree_make_string_view((const char*)storage, device_path_storage.size());
  }
  *out_power_mode = to_c_power_mode(native_power_mode);
  return iree_ok_status();
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  return iree_hal_amdxdna_native_device_create(options, host_allocator,
                                               out_device);
}

extern "C" void iree_hal_amdxdna_native_device_c_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_destroy(device);
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_power_mode_t power_mode) {
  return iree_hal_amdxdna_native_device_set_power_mode(
      device, from_c_power_mode(power_mode));
}

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_device_caps_t* out_caps) {
  iree_hal_amdxdna_native_device_caps_t caps;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_query_caps(device,
                                                                &caps));
  *out_caps = to_c_device_caps(caps);
  return iree_ok_status();
}
