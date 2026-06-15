// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/device.h"

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <string>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amdxdna/allocator.h"
#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/context_cache.h"
#include "iree/hal/drivers/amdxdna/device_internal.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"
#include "iree/hal/drivers/amdxdna/event.h"
#include "iree/hal/drivers/amdxdna/nop_executable_cache.h"
#include "iree/hal/drivers/amdxdna/semaphore.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/deferred_work_queue.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"
#include "iree/hal/utils/resource_set.h"

#define ARENA_BLOCK_SIZE (32 * 1024)

namespace {
extern const iree_hal_device_vtable_t iree_hal_amdxdna_device_vtable;

static iree_hal_amdxdna_native_c_command_opcode_t
iree_hal_amdxdna_native_command_opcode_to_c(
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

static iree_hal_amdxdna_native_c_buffer_sync_model_t
iree_hal_amdxdna_native_buffer_sync_model_to_c(
    iree_hal_amdxdna_native_buffer_sync_model_t model) {
  switch (model) {
    case iree_hal_amdxdna_native_buffer_sync_model_t::caller_syncs_bindings:
      return IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_CALLER_SYNCS_BINDINGS;
    case iree_hal_amdxdna_native_buffer_sync_model_t::submit_syncs_bindings:
      return IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_SUBMIT_SYNCS_BINDINGS;
  }
  return IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_CALLER_SYNCS_BINDINGS;
}

static iree_hal_amdxdna_native_c_device_caps_t
iree_hal_amdxdna_native_device_caps_to_c(
    const iree_hal_amdxdna_native_device_caps_t& caps) {
  iree_hal_amdxdna_native_c_device_caps_t c_caps = {};
  c_caps.ddi_version = caps.ddi_version;
  c_caps.max_effective_queues = caps.max_effective_queues;
  c_caps.max_command_chain_slots = caps.max_command_chain_slots;
  c_caps.context_image_models = caps.context_image_models;
  c_caps.dispatch_models = caps.dispatch_models;
  c_caps.buffer_sync_model =
      iree_hal_amdxdna_native_buffer_sync_model_to_c(caps.buffer_sync_model);
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
      iree_hal_amdxdna_native_command_opcode_to_c(
          caps.default_dispatch_opcode);
  return c_caps;
}
}  // namespace

iree_hal_amdxdna_device::iree_hal_amdxdna_device(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_TRACE_ZONE_BEGIN(z0);

  channel_provider = nullptr;
  memset(&topology_info, 0, sizeof(topology_info));
  memset(&topology_info_resolved, 0, sizeof(topology_info_resolved));
  proactor_pool = nullptr;
  proactor = nullptr;
  async_queue = nullptr;
  default_slab_provider = nullptr;
  default_pool_notification = nullptr;
  default_pool = nullptr;
  frontier_tracker = nullptr;
  frontier_axis = 0;
  device_allocator = nullptr;
  native_device = nullptr;
  memset(&native_caps, 0, sizeof(native_caps));
  context_cache = iree_hal_amdxdna_device_context_cache_create();
  chain_command_cache = nullptr;
  single_command_cache = nullptr;
  iree_atomic_store(&chain_max_slots, 0, iree_memory_order_relaxed);

  iree_hal_resource_initialize(&iree_hal_amdxdna_device_vtable, &resource);
  this->host_allocator = host_allocator;
  this->power_mode_applied = false;

  iree_arena_block_pool_initialize(ARENA_BLOCK_SIZE, host_allocator,
                                   &block_pool);

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_amdxdna_device::~iree_hal_amdxdna_device() {
  iree_hal_amdxdna_device_destroy_single_command_cache(this);
  iree_hal_amdxdna_device_destroy_chain_command_cache(this);
  iree_hal_amdxdna_device_context_cache_destroy(context_cache);
  context_cache = nullptr;
}

static iree_status_t iree_hal_amdxdna_device_initialize_hal_resources(
    iree_hal_amdxdna_device* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_allocator_create(
      device->host_allocator, device->native_device,
      &device->device_allocator));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_async_queue_create(
      &device->block_pool, device->host_allocator, &device->async_queue));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_initialize_async(
    iree_hal_amdxdna_device* device,
    const iree_hal_device_create_params_t* create_params) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  return iree_async_proactor_pool_get(device->proactor_pool, 0,
                                      &device->proactor);
}

// Creates the device's default slab-backed pool. Pattern mirrors
// iree_hal_sync_device_create_default_pool: CPU slab provider + proactor-
// backed notification + passthrough pool wrapping them. The slab provider
// owns CPU-side bookkeeping; amdxdna still allocates real BOs for explicit
// alloca, but the slab/notification pair is what the CTS Explicit*Pool* tests
// require to drive their pool epochs.
static iree_status_t iree_hal_amdxdna_device_create_default_pool(
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_slab_provider,
    iree_async_notification_t** out_notification, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(proactor);
  *out_slab_provider = nullptr;
  *out_notification = nullptr;
  *out_pool = nullptr;

  iree_hal_slab_provider_t* slab_provider = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_hal_cpu_slab_provider_create(host_allocator, &slab_provider));

  iree_async_notification_t* notification = nullptr;
  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification);
  if (iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_options_t options = {0};
    status = iree_hal_passthrough_pool_create(
        options, slab_provider, notification, host_allocator, out_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_provider = slab_provider;
    *out_notification = notification;
    slab_provider = nullptr;
    notification = nullptr;
  }
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
  return status;
}

// Pool epoch query callback. The frontier_tracker is observed via the queue's
// per-axis epoch counter; when this returns true the test sees pool progress.
static bool iree_hal_amdxdna_device_query_pool_epoch(void* user_data,
                                                     iree_async_axis_t axis,
                                                     uint64_t epoch) {
  iree_async_frontier_tracker_t* tracker =
      reinterpret_cast<iree_async_frontier_tracker_t*>(user_data);
  if (!tracker) return false;
  return iree_async_frontier_tracker_query_epoch(tracker, axis, epoch);
}

static iree_status_t iree_hal_amdxdna_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  (void)queue_affinity;
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (!device->default_slab_provider || !device->default_pool_notification) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna default pool not initialized");
  }
  out_backend->slab_provider = device->default_slab_provider;
  out_backend->notification = device->default_pool_notification;
  out_backend->epoch_query = iree_hal_pool_epoch_query_t{
      iree_hal_amdxdna_device_query_pool_epoch, device->frontier_tracker};
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (!topology_info) {
    if (device->frontier_tracker) {
      iree_async_frontier_tracker_retire_axis(
          device->frontier_tracker, device->frontier_axis,
          iree_status_from_code(IREE_STATUS_CANCELLED));
      iree_hal_amdxdna_async_queue_set_frontier(device->async_queue, nullptr,
                                                0);
      iree_async_frontier_tracker_release(device->frontier_tracker);
      device->frontier_tracker = nullptr;
      device->frontier_axis = 0;
    }
    memset(&device->topology_info, 0, sizeof(device->topology_info));
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* tracker = topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  if (!tracker) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "topology frontier tracker must be provided");
  }
  if (device->frontier_tracker) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna topology info is already assigned; "
                            "clear it before assigning a new topology");
  }
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      tracker, axis, /*semaphore=*/nullptr));
  device->topology_info = *topology_info;
  device->frontier_tracker = tracker;
  device->frontier_axis = axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  iree_hal_amdxdna_async_queue_set_frontier(device->async_queue, tracker, axis);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_nop_executable_cache_create(
      device->native_device, identifier, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_amdxdna_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_deferred_command_buffer_create(
      device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, &device->block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_hal_semaphore_compatibility_t
iree_hal_amdxdna_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  (void)base_device;
  (void)semaphore;
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_amdxdna_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_semaphore_create(
      device->proactor, queue_affinity, initial_value, flags,
      device->host_allocator, out_semaphore);
}

static iree_status_t iree_hal_amdxdna_validate_execute_flags(
    iree_hal_execute_flags_t flags) {
  const iree_hal_execute_flags_t supported_flags =
      IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported execute flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_create_binding_table_resource_set(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags,
    iree_hal_resource_set_t** out_resource_set) {
  *out_resource_set = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_execute_flags(flags));
  if (!command_buffer || command_buffer->binding_count == 0 ||
      iree_any_bit_set(flags,
                       IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(binding_table.count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer requires at least %u "
                            "bindings but no binding table was provided",
                            command_buffer->binding_count);
  }
  if (IREE_UNLIKELY(binding_table.count < command_buffer->binding_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect command buffer requires at least %u bindings but only "
        "%" PRIhsz " were provided",
        command_buffer->binding_count, binding_table.count);
  }
  if (IREE_UNLIKELY(!binding_table.bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer binding table storage is "
                            "NULL for %" PRIhsz " bindings",
                            binding_table.count);
  }

  iree_hal_resource_set_t* resource_set = nullptr;
  iree_status_t status =
      iree_hal_resource_set_allocate(&device->block_pool, &resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert_strided(
        resource_set, command_buffer->binding_count, binding_table.bindings,
        offsetof(iree_hal_buffer_binding_t, buffer),
        sizeof(iree_hal_buffer_binding_t));
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(resource_set);
    *out_resource_set = resource_set;
  } else {
    iree_hal_resource_set_free(resource_set);
  }
  return status;
}

struct iree_hal_amdxdna_queue_execute_op_t {
  iree_hal_amdxdna_device* device;
  iree_allocator_t host_allocator;
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t binding_table;
  iree_hal_resource_set_t* binding_resource_set;
};

static void iree_hal_amdxdna_queue_execute_op_cleanup(void* user_data) {
  auto* op = reinterpret_cast<iree_hal_amdxdna_queue_execute_op_t*>(user_data);
  if (!op) return;
  iree_hal_resource_set_free(op->binding_resource_set);
  iree_hal_command_buffer_release(op->command_buffer);
  iree_allocator_free(op->host_allocator, op);
}

static iree_status_t iree_hal_amdxdna_validate_queue_execute_binding_table(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_binding_table_t* out_binding_table) {
  if (IREE_UNLIKELY(!command_buffer && binding_table.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }

  const iree_host_size_t binding_count =
      command_buffer ? command_buffer->binding_count : 0;
  if (binding_count == 0) {
    *out_binding_table = iree_hal_buffer_binding_table_empty();
  } else if (IREE_UNLIKELY(binding_table.count < binding_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "indirect command buffer requires at least %" PRIhsz
                            " bindings but only %" PRIhsz " were provided",
                            binding_count, binding_table.count);
  } else if (IREE_UNLIKELY(!binding_table.bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer binding table storage is "
                            "NULL for %" PRIhsz " bindings",
                            binding_table.count);
  } else {
    *out_binding_table =
        iree_hal_buffer_binding_table_t{binding_count, binding_table.bindings};
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_queue_execute_op_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags,
    iree_hal_amdxdna_queue_execute_op_t** out_op) {
  *out_op = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_execute_flags(flags));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_queue_execute_binding_table(
      command_buffer, binding_table, &binding_table));
  const iree_host_size_t binding_count = binding_table.count;

  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
          sizeof(iree_hal_amdxdna_queue_execute_op_t), binding_count,
          sizeof(iree_hal_buffer_binding_t), &total_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue_execute binding table copy is too large");
  }

  iree_hal_amdxdna_queue_execute_op_t* op = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(device->host_allocator, total_size,
                                             reinterpret_cast<void**>(&op)));
  memset(op, 0, total_size);
  op->device = device;
  op->host_allocator = device->host_allocator;
  op->command_buffer = command_buffer;
  iree_hal_command_buffer_retain(command_buffer);

  iree_status_t status = iree_hal_amdxdna_create_binding_table_resource_set(
      device, command_buffer, binding_table, flags, &op->binding_resource_set);
  if (iree_status_is_ok(status) && binding_count > 0) {
    auto* bindings_copy = reinterpret_cast<iree_hal_buffer_binding_t*>(
        reinterpret_cast<uint8_t*>(op) +
        sizeof(iree_hal_amdxdna_queue_execute_op_t));
    memcpy(bindings_copy, binding_table.bindings,
           binding_count * sizeof(*binding_table.bindings));
    op->binding_table.count = binding_count;
    op->binding_table.bindings = bindings_copy;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_queue_execute_op_cleanup(op);
    return status;
  }

  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_queue_execute_apply(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  if (!command_buffer) return iree_ok_status();
  iree_hal_command_buffer_t* direct_command_buffer = nullptr;
  iree_hal_command_buffer_mode_t mode =
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
      IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
  iree_status_t status = iree_hal_amdxdna_direct_command_buffer_create(
      device, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
      /*binding_capacity=*/0, &device->block_pool, device->host_allocator,
      &direct_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_deferred_command_buffer_apply(
        command_buffer, direct_command_buffer, binding_table);
  }
  iree_hal_command_buffer_release(direct_command_buffer);
  return status;
}

static iree_status_t iree_hal_amdxdna_queue_execute_op_fn(void* user_data) {
  auto* op = reinterpret_cast<iree_hal_amdxdna_queue_execute_op_t*>(user_data);
  return iree_hal_amdxdna_queue_execute_apply(op->device, op->command_buffer,
                                              op->binding_table);
}

static iree_status_t iree_hal_amdxdna_complete_queue_op(
    iree_hal_amdxdna_device* device,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/nullptr);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_async_queue_advance_frontier(device->async_queue);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

static bool iree_hal_amdxdna_wait_list_is_ready(
    const iree_hal_semaphore_list_t wait_semaphore_list) {
  return iree_hal_semaphore_list_is_empty(wait_semaphore_list) ||
         iree_hal_semaphore_list_poll(wait_semaphore_list);
}

static iree_status_t iree_hal_amdxdna_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  iree_hal_amdxdna_queue_execute_op_t* op = nullptr;
  bool signal_list_resolved = false;
  iree_status_t status = iree_hal_amdxdna_validate_execute_flags(flags);
  iree_hal_buffer_binding_table_t validated_binding_table =
      iree_hal_buffer_binding_table_empty();
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_validate_queue_execute_binding_table(
        command_buffer, binding_table, &validated_binding_table);
  }

  const bool wait_list_ready =
      iree_status_is_ok(status) &&
      iree_hal_amdxdna_wait_list_is_ready(wait_semaphore_list);
  if (wait_list_ready) {
    status = iree_hal_amdxdna_queue_execute_apply(device, command_buffer,
                                                  validated_binding_table);
    status = iree_hal_amdxdna_complete_queue_op(device, signal_semaphore_list,
                                                status);
    signal_list_resolved = true;
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_queue_execute_op_create(
        device, command_buffer, validated_binding_table, flags, &op);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_async_queue_enqueue(
          device->async_queue, wait_semaphore_list, signal_semaphore_list,
          iree_hal_amdxdna_queue_execute_op_fn,
          iree_hal_amdxdna_queue_execute_op_cleanup, op,
          /*retained_resources=*/nullptr, /*retained_resource_count=*/0);
      if (iree_status_is_ok(status)) op = nullptr;  // async queue owns op.
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_queue_execute_op_cleanup(op);
    if (!signal_list_resolved) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdxdna_device_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_t* old_allocator = device->device_allocator;
  device->device_allocator = new_allocator;
  iree_hal_allocator_release(old_allocator);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdxdna_device_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_amdxdna_device_trim(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  // Keep this conservative. amdxdna semaphores are host-side, BOs require
  // explicit sync, and this single-device backend does not expose peer, SVA,
  // or hardware timeline semaphore guarantees.
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_amdxdna_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_amdxdna_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  (void)base_device;
  (void)queue_affinity;
  (void)params;
  (void)out_channel;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_amdxdna_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  return iree_hal_amdxdna_event_create(queue_affinity, flags,
                                       device->host_allocator, out_event);
}

static iree_status_t iree_hal_amdxdna_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  (void)flags;
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      /*proactor=*/nullptr, iree_hal_device_host_allocator(base_device),
      out_file);
}

static iree_status_t iree_hal_amdxdna_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_value = 0;
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value =
        (iree_string_view_equal(key, IREE_SV("amdxdna-pdi-fb")) ||
         iree_string_view_equal(key, IREE_SV("amdxdna-xclbin-fb")) ||
         iree_string_view_equal(key, IREE_SV("amdaie-amdxdna-xclbin-fb")))
            ? 1
            : 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported query");
}

static iree_status_t iree_hal_amdxdna_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)pool;
  (void)flags;

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  // Allocate the buffer synchronously: the caller needs it returned now even
  // though the wait_semaphore_list may not yet be satisfied. This is the
  // standard "async placement" contract: the buffer object exists, but
  // signal_semaphore_list will not fire until the waits are resolved (which
  // is what the deferred async_queue task below ensures).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(device->device_allocator, params,
                                             allocation_size, out_buffer));
  iree_hal_amdxdna_buffer_mark_allocated(
      iree_hal_buffer_allocated_buffer(*out_buffer));

  // Tag the buffer with async placement so callers can route dealloca back to
  // this device/queue.
  //
  // The CTS BufferMetadata test asserts placement.queue_affinity has exactly
  // one bit set (so callers can use it as a target queue id directly). The
  // caller passes IREE_HAL_QUEUE_AFFINITY_ANY which is all-bits, so we
  // collapse to a single bit here. Today amdxdna reports a single physical
  // queue; the lowest-set-bit pick is therefore both deterministic and
  // correct. A multi-queue redesign would need a real scheduling decision
  // here instead of a fixed pick.
  // Once amdxdna reports more than one HAL queue this should become a
  // queue-affinity -> ordinal lookup/scheduling decision.
  iree_hal_queue_affinity_t selected =
      iree_hal_queue_affinity_is_empty(queue_affinity)
          ? iree_hal_queue_affinity_t{1}
          : iree_hal_queue_affinity_t{
                1ull << iree_hal_queue_affinity_find_first_set(queue_affinity)};
  (*out_buffer)->placement = iree_hal_buffer_placement_t{
      base_device, selected, IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS, 0};

  // Defer signaling until wait_semaphore_list is satisfied. The op body is
  // empty; the buffer is already usable from the caller's POV; we just need
  // to fire signal_semaphore_list at the right time. If the wait list is
  // already satisfied, complete inline to avoid bouncing hot allocations
  // through the worker thread and making the following dispatch wait pending.
  // No resources to retain (the buffer is owned by the caller, not by the
  // deferred signal task).
  iree_status_t completion_status = iree_ok_status();
  if (iree_hal_amdxdna_wait_list_is_ready(wait_semaphore_list)) {
    completion_status = iree_hal_amdxdna_complete_queue_op(
        device, signal_semaphore_list, iree_ok_status());
  } else {
    completion_status = iree_hal_amdxdna_async_queue_enqueue(
        device->async_queue, wait_semaphore_list, signal_semaphore_list,
        /*op_fn=*/nullptr, /*cleanup_fn=*/nullptr, /*user_data=*/nullptr,
        /*retained_resources=*/nullptr, /*retained_resource_count=*/0);
  }
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_buffer_release(*out_buffer);
    *out_buffer = nullptr;
    IREE_TRACE_ZONE_END(z0);
    return completion_status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// op_fn that marks a buffer as deallocated. Runs on the async_queue worker
// when the op fires successfully. user_data is a borrowed buffer pointer;
// the buffer's lifetime is owned by the queue's retained_resources mechanism
// (separately retained by queue_dealloca below) so we do NOT release here.
// If the op is cancelled, the queue still releases the retained buffer; we
// just don't get to mark it deallocated, which is fine because the buffer
// is on its way out anyway.
static iree_status_t iree_hal_amdxdna_dealloca_op_fn(void* user_data) {
  iree_hal_buffer_t* buffer = reinterpret_cast<iree_hal_buffer_t*>(user_data);
  iree_hal_amdxdna_buffer_mark_deallocated(buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_validate_live_buffer(
    iree_hal_buffer_t* buffer, const char* role) {
  if (iree_hal_amdxdna_buffer_is_deallocated(
          iree_hal_buffer_allocated_buffer(buffer))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is a deallocated amdxdna buffer", role);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  if (iree_hal_amdxdna_wait_list_is_ready(wait_semaphore_list)) {
    iree_hal_amdxdna_buffer_mark_deallocated(buffer);
    return iree_hal_amdxdna_complete_queue_op(device, signal_semaphore_list,
                                              iree_ok_status());
  }

  // Retain the buffer and hand it to the queue as a retained_resource. The
  // queue will release it on every termination path (success, cancellation,
  // wait failure), so the +1 retain is never leaked even if op_fn doesn't
  // run. user_data is the same buffer pointer (borrowed) for op_fn to use
  // while the retain is held.
  iree_hal_buffer_retain(buffer);
  iree_hal_resource_t* retained[] = {
      reinterpret_cast<iree_hal_resource_t*>(buffer),
  };
  iree_status_t status = iree_hal_amdxdna_async_queue_enqueue(
      device->async_queue, wait_semaphore_list, signal_semaphore_list,
      iree_hal_amdxdna_dealloca_op_fn, /*cleanup_fn=*/nullptr,
      /*user_data=*/buffer,
      /*retained_resources=*/retained,
      /*retained_resource_count=*/IREE_ARRAYSIZE(retained));
  if (!iree_status_is_ok(status)) {
    // Enqueue failed; caller still owns the +1 retain.
    iree_hal_buffer_release(buffer);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      target_buffer, "queue_fill target"));
  return iree_hal_device_queue_emulated_fill(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      target_buffer, target_offset, length, pattern, pattern_length, flags);
}

static iree_status_t iree_hal_amdxdna_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      target_buffer, "queue_update target"));
  return iree_hal_device_queue_emulated_update(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_buffer, target_offset, length,
      flags);
}

struct iree_hal_amdxdna_queue_copy_op_t {
  iree_allocator_t host_allocator;
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
};

static iree_status_t iree_hal_amdxdna_validate_copy(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      source_buffer, "queue_copy source"));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      target_buffer, "queue_copy target"));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));
  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(iree_hal_buffer_test_overlap(source_buffer, source_offset,
                                                 length, target_buffer,
                                                 target_offset, length) !=
                    IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_buffer_ranges(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_buffer_t* allocated_source_buffer =
      iree_hal_buffer_allocated_buffer(source_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      allocated_source_buffer, "queue_copy source"));
  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      allocated_target_buffer, "queue_copy target"));
  if (length == 0) return iree_ok_status();

  iree_hal_amdxdna_native_buffer_t* source_device_buffer =
      iree_hal_amdxdna_buffer_handle(allocated_source_buffer);
  void* source_device_buffer_ptr = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(
      source_device_buffer, &source_device_buffer_ptr));
  iree_device_size_t source_absolute_offset =
      iree_hal_buffer_byte_offset(source_buffer) + source_offset;

  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(allocated_target_buffer);
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(
      target_device_buffer, &target_device_buffer_ptr));
  iree_device_size_t target_absolute_offset =
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync(
      source_device_buffer,
      iree_hal_amdxdna_native_sync_direction_t::device_to_host, length,
      source_absolute_offset));
  memcpy(reinterpret_cast<uint8_t*>(target_device_buffer_ptr) +
             target_absolute_offset,
         reinterpret_cast<uint8_t*>(source_device_buffer_ptr) +
             source_absolute_offset,
         length);
  return iree_hal_amdxdna_native_buffer_sync(
      target_device_buffer,
      iree_hal_amdxdna_native_sync_direction_t::host_to_device, length,
      target_absolute_offset);
}

static iree_status_t iree_hal_amdxdna_queue_copy_op_fn(void* user_data) {
  auto* op = reinterpret_cast<iree_hal_amdxdna_queue_copy_op_t*>(user_data);
  return iree_hal_amdxdna_copy_buffer_ranges(
      op->source_buffer, op->source_offset, op->target_buffer,
      op->target_offset, op->length);
}

static void iree_hal_amdxdna_queue_copy_op_cleanup(void* user_data) {
  auto* op = reinterpret_cast<iree_hal_amdxdna_queue_copy_op_t*>(user_data);
  if (!op) return;
  iree_allocator_free(op->host_allocator, op);
}

static iree_status_t iree_hal_amdxdna_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  (void)queue_affinity;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_copy(
      source_buffer, source_offset, target_buffer, target_offset, length,
      flags));

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  if (iree_hal_amdxdna_wait_list_is_ready(wait_semaphore_list)) {
    iree_status_t status = iree_hal_amdxdna_copy_buffer_ranges(
        source_buffer, source_offset, target_buffer, target_offset, length);
    return iree_hal_amdxdna_complete_queue_op(device, signal_semaphore_list,
                                              status);
  }

  iree_hal_amdxdna_queue_copy_op_t* op = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      device->host_allocator, sizeof(*op), reinterpret_cast<void**>(&op)));
  op->host_allocator = device->host_allocator;
  op->source_buffer = source_buffer;
  op->source_offset = source_offset;
  op->target_buffer = target_buffer;
  op->target_offset = target_offset;
  op->length = length;

  iree_hal_buffer_retain(source_buffer);
  iree_hal_buffer_retain(target_buffer);
  iree_hal_resource_t* retained[] = {
      reinterpret_cast<iree_hal_resource_t*>(source_buffer),
      reinterpret_cast<iree_hal_resource_t*>(target_buffer),
  };
  iree_status_t status = iree_hal_amdxdna_async_queue_enqueue(
      device->async_queue, wait_semaphore_list, signal_semaphore_list,
      iree_hal_amdxdna_queue_copy_op_fn, iree_hal_amdxdna_queue_copy_op_cleanup,
      op,
      /*retained_resources=*/retained,
      /*retained_resource_count=*/IREE_ARRAYSIZE(retained));
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(source_buffer);
    iree_hal_buffer_release(target_buffer);
    iree_hal_amdxdna_queue_copy_op_cleanup(op);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      target_buffer, "queue_read target"));
  iree_hal_file_transfer_options_t options = {
      IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT};
  return iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options);
}

static iree_status_t iree_hal_amdxdna_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_live_buffer(
      source_buffer, "queue_write source"));
  iree_hal_file_transfer_options_t options = {
      IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT};
  return iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options);
}

static iree_status_t iree_hal_amdxdna_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_hal_device_queue_emulated_host_call(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      call, args, flags);
}

static iree_status_t iree_hal_amdxdna_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_hal_device_queue_emulated_dispatch(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      executable, function, config, constants, bindings, flags);
}

static iree_status_t iree_hal_amdxdna_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  (void)base_device;
  (void)queue_affinity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue profiling is not implemented");
}

static iree_status_t iree_hal_amdxdna_device_profiling_flush(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue profiling is not implemented");
}

static iree_status_t iree_hal_amdxdna_device_profiling_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue profiling is not implemented");
}

static iree_string_view_t iree_hal_amdxdna_device_id(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return device->identifier;
}

static void iree_hal_amdxdna_device_destroy(iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  // Drain and shut down the async queue before tearing down the block pool
  // (the queue's pending ops live in arenas backed by the block pool).
  if (device->async_queue) {
    iree_hal_amdxdna_async_queue_destroy(device->async_queue);
    device->async_queue = nullptr;
  }
  // Retire the frontier axis (if assigned) before releasing the tracker.
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->frontier_tracker, device->frontier_axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = nullptr;
  }
  if (device->default_pool) {
    iree_hal_pool_release(device->default_pool);
    device->default_pool = nullptr;
  }
  if (device->default_slab_provider) {
    iree_hal_slab_provider_release(device->default_slab_provider);
    device->default_slab_provider = nullptr;
  }
  if (device->default_pool_notification) {
    iree_async_notification_release(device->default_pool_notification);
    device->default_pool_notification = nullptr;
  }
  iree_arena_block_pool_deinitialize(&device->block_pool);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_allocator_release(device->device_allocator);
  if (device->proactor_pool) {
    iree_async_proactor_pool_release(device->proactor_pool);
  }
  if (device->power_mode_applied && device->native_device) {
    (void)iree_hal_amdxdna_native_device_set_power_mode(
        device->native_device,
        iree_hal_amdxdna_native_power_mode_t::default_mode);
    device->power_mode_applied = false;
  }
  // Drop cached native command BOs before tearing down the native device they
  // were allocated from. The destructor also calls these helpers, so this is
  // safe if an earlier failure path already cleared the caches.
  iree_hal_amdxdna_device_destroy_single_command_cache(device);
  iree_hal_amdxdna_device_destroy_chain_command_cache(device);
  // Drop the cache's shared_ptr<native_context> refs before the native device
  // they reference is torn down. Per the IREE HAL lifetime contract,
  // executables (which co-own these via executable->context) are released
  // before their device, so by the time we get here the cache holds the last
  // refs and clear() runs the native context destructors cleanly.
  // Lock defensively against the contract being violated (zero cost
  // uncontended) and so a future audit can't ask "is this clear racy?"
  iree_hal_amdxdna_device_context_cache_clear(device->context_cache);
  iree_hal_amdxdna_native_device_destroy(device->native_device);
  // The device struct is placement-new'd in iree_hal_amdxdna_device_create;
  // explicitly run the destructor before freeing the storage so non-trivial
  // members and private cache state release cleanly. Save host_allocator first;
  // accessing `device->` after the destructor is UB.
  iree_allocator_t host_allocator = device->host_allocator;
  device->~iree_hal_amdxdna_device();
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
};

iree_hal_amdxdna_device* iree_hal_amdxdna_device_cast(
    iree_hal_device_t* base_device) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
}

static iree_allocator_t iree_hal_amdxdna_device_host_allocator(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_amdxdna_device_device_allocator(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return device->device_allocator;
}

void iree_hal_amdxdna_device_options_initialize(
    iree_hal_amdxdna_device_params* out_options) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_options, 0, sizeof(*out_options));
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdxdna_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdxdna_device_params* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device_params resolved_options;
  std::string resolved_device_path_storage;
  iree_hal_amdxdna_native_power_mode_t resolved_power_mode =
      iree_hal_amdxdna_native_power_mode_t::default_mode;
  bool should_set_power_mode = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_resolve_device_options(
              options, &resolved_options, &resolved_device_path_storage,
              &resolved_power_mode, &should_set_power_mode));

  iree_hal_amdxdna_device* device = nullptr;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                reinterpret_cast<void**>(&device)));
  device =
      new (device) iree_hal_amdxdna_device(&resolved_options, host_allocator);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      reinterpret_cast<char*>(device) + total_size - identifier.size);

  iree_status_t status = iree_ok_status();
  status = iree_hal_amdxdna_native_device_create(
      &resolved_options, device->host_allocator, &device->native_device);
  if (iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_device_caps_t native_caps;
    status = iree_hal_amdxdna_native_device_query_caps(device->native_device,
                                                       &native_caps);
    if (iree_status_is_ok(status)) {
      device->native_caps =
          iree_hal_amdxdna_native_device_caps_to_c(native_caps);
    }
  }
  if (iree_status_is_ok(status) && should_set_power_mode) {
    status = iree_hal_amdxdna_native_device_set_power_mode(
        device->native_device, resolved_power_mode);
    if (iree_status_is_ok(status) &&
        resolved_power_mode !=
            iree_hal_amdxdna_native_power_mode_t::default_mode) {
      device->power_mode_applied = true;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_device_initialize_hal_resources(device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_device_initialize_async(device, create_params);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_device_create_default_pool(
        device->proactor, device->host_allocator,
        &device->default_slab_provider, &device->default_pool_notification,
        &device->default_pool);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_device_destroy(
        reinterpret_cast<iree_hal_device_t*>(device));
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_device = reinterpret_cast<iree_hal_device_t*>(device);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

namespace {
const iree_hal_device_vtable_t iree_hal_amdxdna_device_vtable = {
    iree_hal_amdxdna_device_destroy,
    iree_hal_amdxdna_device_id,
    iree_hal_amdxdna_device_host_allocator,
    iree_hal_amdxdna_device_device_allocator,
    iree_hal_amdxdna_device_replace_device_allocator,
    iree_hal_amdxdna_device_replace_channel_provider,
    iree_hal_amdxdna_device_trim,
    iree_hal_amdxdna_device_query_i64,
    iree_hal_amdxdna_device_query_capabilities,
    iree_hal_amdxdna_device_topology_info,
    iree_hal_amdxdna_device_refine_topology_edge,
    iree_hal_amdxdna_device_assign_topology_info,
    iree_hal_amdxdna_device_create_channel,
    iree_hal_amdxdna_device_create_command_buffer,
    iree_hal_amdxdna_device_create_event,
    iree_hal_amdxdna_device_create_executable_cache,
    iree_hal_amdxdna_device_import_file,
    iree_hal_amdxdna_device_create_semaphore,
    iree_hal_amdxdna_device_query_semaphore_compatibility,
    // Pool-backed queue alloca is not supported. Returning UNIMPLEMENTED here
    // makes pool-using tests fail gracefully instead of NULL-deref crashing
    // through the vtable.
    iree_hal_amdxdna_device_query_queue_pool_backend,
    iree_hal_amdxdna_device_queue_alloca,
    iree_hal_amdxdna_device_queue_dealloca,
    iree_hal_amdxdna_device_queue_fill,
    iree_hal_amdxdna_device_queue_update,
    iree_hal_amdxdna_device_queue_copy,
    iree_hal_amdxdna_device_queue_read,
    iree_hal_amdxdna_device_queue_write,
    iree_hal_amdxdna_device_queue_host_call,
    iree_hal_amdxdna_device_queue_dispatch,
    iree_hal_amdxdna_device_queue_execute,
    iree_hal_amdxdna_device_queue_flush,
    // Returning UNIMPLEMENTED here signals callers (e.g. the CTS profiling
    // tests) to skip profiling-dependent assertions instead of treating the
    // begin as a successful no-op and then failing because no events were
    // recorded.
    iree_hal_amdxdna_device_profiling_begin,
    iree_hal_amdxdna_device_profiling_flush,
    iree_hal_amdxdna_device_profiling_end,
    nullptr,
    nullptr,
};
}
