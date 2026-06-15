// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file is included by exactly one platform native backend translation unit.
// It attaches the public C DDI in native.h to the current private C++ native
// implementation while the backend internals are being converted to C.

#include "iree/hal/drivers/amdxdna/native.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/hal/drivers/amdxdna/native_internal.h"

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

iree_status_t from_c_command_opcode(
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_opcode_t* out_opcode) {
  switch (opcode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU:
      *out_opcode = iree_hal_amdxdna_native_command_opcode_t::start_cu;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU:
      *out_opcode = iree_hal_amdxdna_native_command_opcode_t::start_npu;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF:
      *out_opcode =
          iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN:
      *out_opcode = iree_hal_amdxdna_native_command_opcode_t::command_chain;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown amdxdna native command opcode");
}

iree_status_t from_c_context_image_type(
    iree_hal_amdxdna_native_c_context_image_type_t type,
    iree_hal_amdxdna_native_context_image_type_t* out_type) {
  switch (type) {
    case IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI:
      *out_type = iree_hal_amdxdna_native_context_image_type_t::pdi;
      return iree_ok_status();
    case IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN:
      *out_type = iree_hal_amdxdna_native_context_image_type_t::xclbin;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown amdxdna native context image type");
}

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

struct iree_hal_amdxdna_native_context_ref_t {
  std::shared_ptr<iree_hal_amdxdna_native_context_t> context;
};

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

extern "C" iree_status_t iree_hal_amdxdna_native_device_c_create_context_ref(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_c_context_image_t* image,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  *out_context_ref = nullptr;
  iree_hal_amdxdna_native_context_image_t native_image;
  IREE_RETURN_IF_ERROR(from_c_context_image_type(image->type,
                                                &native_image.type));
  native_image.pdi = image->pdi;
  native_image.xclbin = image->xclbin;
  native_image.kernel_name = image->kernel_name;
  iree_hal_amdxdna_native_context_t* raw_context = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_create_context(
      device, &native_image, &raw_context));
  auto* context_ref = new iree_hal_amdxdna_native_context_ref_t();
  context_ref->context = std::shared_ptr<iree_hal_amdxdna_native_context_t>(
      raw_context, iree_hal_amdxdna_native_context_destroy);
  *out_context_ref = context_ref;
  return iree_ok_status();
}

extern "C" iree_hal_amdxdna_native_context_ref_t*
iree_hal_amdxdna_native_context_ref_retain(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  if (!context_ref) return nullptr;
  auto* retained = new iree_hal_amdxdna_native_context_ref_t();
  retained->context = context_ref->context;
  return retained;
}

extern "C" void iree_hal_amdxdna_native_context_ref_release(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  delete context_ref;
}

extern "C" iree_hal_amdxdna_native_context_t*
iree_hal_amdxdna_native_context_ref_borrow(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  return context_ref ? context_ref->context.get() : nullptr;
}

extern "C" iree_status_t iree_hal_amdxdna_native_context_ref_open_cu(
    iree_hal_amdxdna_native_context_ref_t* context_ref,
    iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_c_cu_index_t* out_cu_index) {
  iree_hal_amdxdna_native_cu_index_t native_cu_index;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_context_open_cu(
      context_ref->context.get(), kernel_name, &native_cu_index));
  out_cu_index->index = native_cu_index.index;
  return iree_ok_status();
}

extern "C" iree_status_t
iree_hal_amdxdna_native_context_ref_close_single_aperture_session(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  return iree_hal_amdxdna_native_context_close_single_aperture_session(
      context_ref->context.get());
}

extern "C" iree_hal_amdxdna_native_queue_t*
iree_hal_amdxdna_native_context_ref_queue(
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  return iree_hal_amdxdna_native_context_queue(context_ref->context.get());
}

extern "C" uint64_t iree_hal_amdxdna_native_queue_c_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return iree_hal_amdxdna_native_queue_exec_command_count(queue);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_device_c_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  return iree_hal_amdxdna_native_device_query_chain_max_slots(device,
                                                             out_max_slots);
}

extern "C" iree_host_size_t
iree_hal_amdxdna_native_command_c_arg_binding_capacity(void) {
  return iree_hal_amdxdna_native_command_arg_binding_capacity();
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_c_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_t** out_command) {
  *out_command = nullptr;
  iree_hal_amdxdna_native_command_opcode_t native_opcode;
  IREE_RETURN_IF_ERROR(from_c_command_opcode(opcode, &native_opcode));
  iree_hal_amdxdna_native_command_ptr command;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_command_create(device, native_opcode, &command));
  *out_command = command.release();
  return iree_ok_status();
}

extern "C" void iree_hal_amdxdna_native_command_c_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  iree_hal_amdxdna_native_command_destroy(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_c_cu_index_t cu_index) {
  iree_hal_amdxdna_native_cu_index_t native_cu_index;
  native_cu_index.index = cu_index.index;
  return iree_hal_amdxdna_native_command_set_cu_index(command,
                                                     native_cu_index);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_command_c_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  return iree_hal_amdxdna_native_command_add_control_buffer(
      command, control_buffer, control_buffer_size);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  return iree_hal_amdxdna_native_command_add_arg_32(command, value);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  return iree_hal_amdxdna_native_command_add_arg_64(command, value);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_command_add_buffer_arg(command, buffer);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  return iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
      command, buffer, offset);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, iree_host_size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  return iree_hal_amdxdna_native_command_bind_buffer(command, position, buffer,
                                                    offset, size);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_command_c_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_reset_bound_buffers(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_mark_code_dirty(command);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_command_c_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  return iree_hal_amdxdna_native_command_mark_chain_code_dirty(command);
}

extern "C" iree_status_t iree_hal_amdxdna_native_command_c_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  return iree_hal_amdxdna_native_command_prepare_chain(command, commands,
                                                      command_count);
}

extern "C" iree_status_t iree_hal_amdxdna_native_queue_c_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  return iree_hal_amdxdna_native_queue_submit_and_wait(queue, command, label);
}

extern "C" iree_status_t
iree_hal_amdxdna_native_queue_c_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  return iree_hal_amdxdna_native_queue_submit_all_and_wait(
      queue, commands, command_count, label);
}
