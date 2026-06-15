// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/device_internal.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/utils/resource_set.h"

static const uint64_t kAmdxdnaControlCodeOpcode = 3u;

static bool iree_hal_amdxdna_patch_table_is_valid(
    const iree_hal_amdxdna_u32_list_t* patch_table) {
  return patch_table && patch_table->count != 0 &&
         (patch_table->count % 3 == 0);
}

typedef struct iree_hal_amdxdna_direct_command_buffer {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // A resource set to maintain references to all resources used within this
  // one-shot command buffer.
  iree_hal_resource_set_t* resource_set;
  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;

  iree_hal_amdxdna_device* device;

  // Dispatches that can be lowered through the host-patched partial-ELF path
  // accumulate here until end(). A single child is submitted directly; two or
  // more children, or a multi-control-code/reconfiguration artifact, flush as
  // ERT_CMD_CHAIN(s).
  iree_hal_amdxdna_chain_accum_t chain_accum;
} iree_hal_amdxdna_direct_command_buffer;

static iree_status_t iree_hal_amdxdna_validate_live_dispatch_bindings(
    iree_hal_buffer_ref_list_t bindings) {
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (IREE_UNLIKELY(!bindings.values || !bindings.values[i].buffer)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch binding %" PRIhsz " is NULL", i);
    }
    iree_hal_buffer_t* allocated_buffer =
        iree_hal_buffer_allocated_buffer(bindings.values[i].buffer);
    if (iree_hal_amdxdna_buffer_is_deallocated(allocated_buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding %" PRIhsz " is a deallocated amdxdna buffer", i);
    }
  }
  return iree_ok_status();
}

extern const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable;

iree_status_t iree_hal_amdxdna_direct_command_buffer_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  if (binding_capacity > 0) {
    // Indirect command buffers with binding tables are not supported by this
    // direct recording path.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }
  // The amdxdna CB has no replayable state: begin/end are not implemented as
  // resets, and deferred dispatch groups are finalized by end(). A
  // non-ONE_SHOT CB would carry that state across replays. Require ONE_SHOT to
  // match the only mode IREE creates through us today (queue_execute passes
  // ONE_SHOT | ALLOW_INLINE_EXECUTION | UNVALIDATED) and to fail loudly if a
  // future caller hands us a reusable CB.
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "amdxdna command buffers require "
                            "IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT");
  }
  if (iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna command buffers require retained resource lifetimes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_hal_command_buffer_initialize(
      device->device_allocator, mode, command_categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
      (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_amdxdna_direct_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device = device;
  iree_hal_amdxdna_chain_accum_initialize(&command_buffer->chain_accum);
  iree_arena_initialize(block_pool, &command_buffer->arena);
  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);

  return status;
}

static void iree_hal_amdxdna_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_hal_amdxdna_chain_accum_deinitialize(host_allocator,
                                            &command_buffer->chain_accum);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  // Command buffers are one-shot; create initializes all per-recording state
  // and end() flushes any accumulated chain commands.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)base_command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)memory_barrier_count;
  (void)memory_barriers;

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    const iree_hal_buffer_barrier_t* barrier = &buffer_barriers[i];
    const bool flush_host_to_device =
        iree_any_bit_set(barrier->source_scope,
                         IREE_HAL_ACCESS_SCOPE_HOST_WRITE |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE) &&
        iree_any_bit_set(barrier->target_scope,
                         IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
                             IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                             IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_READ);
    const bool invalidate_device_to_host =
        iree_any_bit_set(barrier->source_scope,
                         IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                             IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE) &&
        iree_any_bit_set(barrier->target_scope,
                         IREE_HAL_ACCESS_SCOPE_HOST_READ |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_READ);
    if (!flush_host_to_device && !invalidate_device_to_host) {
      continue;
    }
    if (IREE_UNLIKELY(!barrier->buffer_ref.buffer)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna direct command buffer cannot sync an indirect buffer "
          "barrier without a resolved buffer");
    }
    if (flush_host_to_device) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_buffer_flush_range(
          barrier->buffer_ref.buffer, barrier->buffer_ref.offset,
          barrier->buffer_ref.length));
    }
    if (invalidate_device_to_host) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_buffer_invalidate_range(
          barrier->buffer_ref.buffer, barrier->buffer_ref.offset,
          barrier->buffer_ref.length));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  // The amdxdna direct command buffer executes synchronously against a single
  // in-order queue today, so recording an event signal has no extra device work
  // to enqueue.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)event_count;
  (void)events;
  return iree_hal_amdxdna_direct_command_buffer_execution_barrier(
      base_command_buffer, source_stage_mask, target_stage_mask,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, memory_barrier_count,
      memory_barriers, buffer_barrier_count, buffer_barriers);
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  // No need to allocate scratch space (in an arena) as the memcpy
  // used below is expected to be synchronized.
  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_map(target_device_buffer,
                                               &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint8_t* dst = (uint8_t*)target_device_buffer_ptr + target_offset;
  memcpy(dst, src, target_ref.length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_sync(
              target_device_buffer,
              IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE,
              target_ref.length, target_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_map(target_device_buffer,
                                               &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint8_t* dst = (uint8_t*)target_device_buffer_ptr + target_offset;
  const iree_device_size_t length = target_ref.length;

  // Fast path for byte-pattern fills (most common case).
  if (pattern_length == 1) {
    memset(dst, *(const uint8_t*)pattern, length);
  } else {
    const uint8_t* p = (const uint8_t*)pattern;
    for (iree_device_size_t i = 0; i < length; ++i) {
      dst[i] = p[i % pattern_length];
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_sync(
              target_device_buffer,
              IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE,
              target_ref.length, target_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_map(target_device_buffer,
                                               &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_hal_amdxdna_native_buffer_t* source_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(source_ref.buffer));
  void* source_device_buffer_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_map(source_device_buffer,
                                               &source_device_buffer_ptr));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;

  // Sync the host-mapped source range so the host memcpy reads device-written
  // data, then sync the target range back to device so a subsequent dispatch
  // sees the freshly copied bytes.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_sync(
              source_device_buffer,
              IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_DEVICE_TO_HOST,
              target_ref.length, source_offset));
  memcpy((uint8_t*)target_device_buffer_ptr + target_offset,
         (uint8_t*)source_device_buffer_ptr + source_offset, target_ref.length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_c_sync(
              target_device_buffer,
              IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE,
              target_ref.length, target_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// ===========================================================================
// ERT_CMD_CHAIN support.
//
// Command buffers are lowered by command-stream shape: one recorded child uses
// the direct single-dispatch native command path, while two or more children,
// or artifacts with reconfiguration/multiple control codes, are batched into an
// ERT_CMD_CHAIN submitted with one issue/wait. Each chain slot is submitted as
// ERT_START_NPU (PARTIAL_ELF) with arg[0]=AIE2_EXEC_BUFFER_KERNEL_OP_TXN so the
// firmware runs the same XAie TXN control code as the direct path; the I/O
// addresses that the CU path lets the firmware patch are instead host-patched
// into the control-code BD registers here.
// ===========================================================================
// TXN-interpreter selector: tells the firmware to interpret the instruction
// buffer as an XAie transaction (same value the default ERT_START_CU path
// passes as its opcode arg).
static const uint32_t kAie2ExecBufferKernelOpTxn = 3;

iree_status_t iree_hal_amdxdna_make_npu_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    const iree_hal_amdxdna_u32_list_t* txn,
    const iree_hal_amdxdna_u32_list_t* patches, const uint64_t* args,
    iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
    const iree_device_size_t* arg_offsets,
    const iree_device_size_t* arg_lengths, size_t arg_count,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    iree_hal_amdxdna_chain_cmd_t* out_cmd) {
  size_t bytes = txn->count * sizeof(uint32_t);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_c_alloc_buffer(
      command_buffer->device->native_device, bytes,
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &out_cmd->ctrl_code));
  void* mapped_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_buffer_c_map(out_cmd->ctrl_code, &mapped_ptr));
  uint32_t* dst = (uint32_t*)mapped_ptr;
  memcpy(dst, txn->data, bytes);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_patch_write32_constants(dst, txn->count, constants));
  if (!iree_hal_amdxdna_apply_patch_table(dst, txn->count, patches->data,
                                          patches->count, args, arg_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna cmd-chain: invalid host patch table for control code");
  }
  uint64_t* binding_device_addrs = NULL;
  if (arg_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        command_buffer->host_allocator, arg_count,
        sizeof(*binding_device_addrs), (void**)&binding_device_addrs));
  }
  for (size_t i = 0; i < arg_count; ++i) {
    binding_device_addrs[i] =
        iree_hal_amdxdna_native_buffer_c_device_address(arg_buffers[i]) +
        arg_offsets[i];
  }
  iree_status_t status = iree_hal_amdxdna_chain_cmd_set_signature(
      command_buffer->host_allocator, out_cmd, dst, txn->count, arg_buffers,
      binding_device_addrs, arg_offsets, arg_lengths, arg_count);
  iree_allocator_free(command_buffer->host_allocator, binding_device_addrs);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_sync_all(
      out_cmd->ctrl_code, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
  const iree_hal_amdxdna_native_c_command_opcode_t command_opcode =
      use_native_partial_elf
          ? IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF
          : IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_create(
      command_buffer->device->native_device, command_opcode,
      &out_cmd->command));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_command_c_set_cu_index(out_cmd->command, cu_idx));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_control_buffer(
      out_cmd->command, out_cmd->ctrl_code, bytes));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_arg_32(
      out_cmd->command, kAie2ExecBufferKernelOpTxn));
  const bool native_uses_dpu_regmap_args =
      !use_native_partial_elf &&
      command_buffer->device->native_caps.default_dispatch_opcode ==
          IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  if (command_opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF) {
    if (IREE_UNLIKELY(arg_count &&
                      (!arg_buffers || !arg_offsets || !arg_lengths))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna PARTIAL_ELF cmd-chain child is missing BO "
          "bindings for its runtime args");
    }
    for (size_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_bind_buffer(
          out_cmd->command, /*position=*/i + 1, arg_buffers[i], arg_offsets[i],
          arg_lengths[i]));
    }
  } else if (native_uses_dpu_regmap_args) {
    // Some native drivers expose DPU kernels through an xclbin XML register
    // map. In that path the runtime data VAs are regular ERT args.
    for (size_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_arg_64(
          out_cmd->command, args[i]));
    }
  }
  out_cmd->built = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group_t* group,
    iree_hal_amdxdna_kernel_params_t* kernel_params, size_t run_idx,
    const uint64_t* args, iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
    const iree_device_size_t* arg_offsets,
    const iree_device_size_t* arg_lengths, size_t arg_count,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    bool defer_build) {
  if (defer_build && group->cmd_count != 0 &&
      iree_hal_amdxdna_chain_cmd_matches_raw_descriptor(
          &group->cmds[group->cmd_count - 1],
          &kernel_params->asm_inst_runlist[run_idx],
          &kernel_params->patch_runlist[run_idx], cu_idx, constants,
          use_native_partial_elf, args, arg_buffers, arg_offsets, arg_lengths,
          arg_count)) {
    if (IREE_UNLIKELY(group->cmds[group->cmd_count - 1].repeat_count ==
                      IREE_HOST_SIZE_MAX)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "amdxdna cmd-chain repeat count overflow");
    }
    ++group->cmds[group->cmd_count - 1].repeat_count;
    return iree_ok_status();
  }
  iree_hal_amdxdna_chain_cmd_t cmd;
  iree_hal_amdxdna_chain_cmd_initialize(&cmd);
  iree_status_t status = iree_hal_amdxdna_chain_cmd_set_deferred_descriptor(
      command_buffer->host_allocator, &cmd,
      &kernel_params->asm_inst_runlist[run_idx],
      &kernel_params->patch_runlist[run_idx], cu_idx, constants,
      use_native_partial_elf, arg_buffers, args, arg_offsets, arg_lengths,
      arg_count);
  if (iree_status_is_ok(status) && !defer_build) {
    status = iree_hal_amdxdna_make_npu_cmd(
        command_buffer, cu_idx, &kernel_params->asm_inst_runlist[run_idx],
        &kernel_params->patch_runlist[run_idx], args, arg_buffers, arg_offsets,
        arg_lengths, arg_count, constants, use_native_partial_elf, &cmd);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_chain_group_append_cmd_move(
        command_buffer->host_allocator, group, &cmd);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_chain_cmd_deinitialize(command_buffer->host_allocator,
                                            &cmd);
  }
  return status;
}

// Accumulate one dispatch's reconfig+exec sub-commands into the command
// buffer's deferred-submit accumulator. Does NOT submit; flush_chains() chooses
// direct single-dispatch submit for one child and ERT_CMD_CHAIN for multi-child
// groups at end(). Dispatches that share a hw queue (e.g. all entry points of
// one control-packet executable, or separate executables resolved to the same
// shared context) accumulate into one group.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_context_ref_t* context_ref,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_hal_amdxdna_kernel_params_t* kernel_params,
    iree_const_byte_span_t constants, bool use_native_partial_elf) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The chained path host-patches I/O addresses using the compiler-emitted
  // patch table (parallel to asm_inst_runlist). Require it: an executable
  // compiled before the patch table existed cannot use cmd-chain.
  iree_status_t status = iree_ok_status();
  uint64_t* binding_addrs = NULL;
  iree_hal_amdxdna_native_buffer_t** binding_buffers = NULL;
  iree_device_size_t* binding_offsets = NULL;
  iree_device_size_t* binding_lengths = NULL;

  if (kernel_params->patch_runlist_count !=
      kernel_params->asm_inst_runlist_count) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna cmd-chain requires a host patch table in the executable "
        "(have %zu patch lists for %zu control codes); recompile with a "
        "patch-table-aware compiler",
        kernel_params->patch_runlist_count,
        kernel_params->asm_inst_runlist_count);
    goto cleanup;
  }

  // Binding device addresses (exec args). For control packets the reconfig arg
  // is the per-reconfiguration data buffer (built below).
  if (bindings.count != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_array(command_buffer->host_allocator,
                                        bindings.count, sizeof(*binding_addrs),
                                        (void**)&binding_addrs));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_array(
                command_buffer->host_allocator, bindings.count,
                sizeof(*binding_buffers), (void**)&binding_buffers));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_array(
                command_buffer->host_allocator, bindings.count,
                sizeof(*binding_offsets), (void**)&binding_offsets));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_array(
                command_buffer->host_allocator, bindings.count,
                sizeof(*binding_lengths), (void**)&binding_lengths));
  }
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    status = iree_hal_amdxdna_native_buffer_c_ensure_allocated(native_buffer);
    if (!iree_status_is_ok(status)) goto cleanup;
    // Match the normal ERT_START_CU path: a binding may reference a subspan of
    // its allocated root BO, so host-patched DDR addresses must include both
    // offsets in addition to the BO base address.
    binding_buffers[j] = native_buffer;
    binding_offsets[j] =
        iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
        bindings.values[j].offset;
    binding_lengths[j] = bindings.values[j].length;
    binding_addrs[j] =
        iree_hal_amdxdna_native_buffer_c_device_address(native_buffer) +
        binding_offsets[j];
  }

  // Append to the current group, opening a new one when the native queue
  // changes (a chain runs on a single native context/queue).
  iree_hal_amdxdna_chain_accum_t* accum = &command_buffer->chain_accum;
  iree_hal_amdxdna_chain_group_t* group = NULL;
  if (accum->group_count != 0) {
    group = &accum->groups[accum->group_count - 1];
  }
  if (!group || group->queue != queue ||
      group->native_partial_elf != use_native_partial_elf) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_chain_accum_append_group(
                command_buffer->host_allocator, accum, &group));
    group->context = iree_hal_amdxdna_native_context_ref_retain(context_ref);
    group->queue = queue;
    group->native_partial_elf = use_native_partial_elf;
  }

  // Defer the per-child native build for the cacheable module-style chain path
  // (partial-ELF, no control-packet reconfiguration). Those children are
  // recorded as lightweight descriptors here and built lazily in flush only on
  // a cache miss, so a steady-state exact hit reuses the cached chain and skips
  // the build. Other paths (reconfiguration, non-partial-ELF) build eagerly.
  const bool defer_build =
      use_native_partial_elf && kernel_params->reconf_data_runlist_count == 0;

  // Emit exactly one kernel slot for this HAL dispatch. Reconfiguration control
  // packets retain their repeat count for existing multi-PDI artifacts.
  size_t num_reconfigurations = kernel_params->reconf_data_runlist_count;
  if (num_reconfigurations == 0) {
    status = iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
        command_buffer, group, kernel_params, /*run_idx=*/0, binding_addrs,
        binding_buffers, binding_offsets, binding_lengths, bindings.count,
        cu_idx, constants, use_native_partial_elf, defer_build);
    if (!iree_status_is_ok(status)) goto cleanup;
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Control-packet data buffer for this reconfiguration (reconfig arg[0]).
      iree_hal_amdxdna_u32_list_t* seq = &kernel_params->reconf_data_runlist[i];
      size_t seq_bytes = seq->count * sizeof(uint32_t);
      iree_hal_amdxdna_native_buffer_t* reconf_buffer = NULL;
      status = iree_hal_amdxdna_native_device_c_alloc_buffer(
          command_buffer->device->native_device, seq_bytes,
          IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY, &reconf_buffer);
      if (!iree_status_is_ok(status)) goto cleanup;
      void* seq_buffer_ptr = NULL;
      status =
          iree_hal_amdxdna_native_buffer_c_map(reconf_buffer, &seq_buffer_ptr);
      if (iree_status_is_ok(status)) {
        memcpy(seq_buffer_ptr, seq->data, seq_bytes);
        status = iree_hal_amdxdna_native_buffer_c_sync_all(
            reconf_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_chain_group_append_reconf_buffer(
            command_buffer->host_allocator, group, reconf_buffer);
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amdxdna_native_buffer_c_destroy(reconf_buffer);
        goto cleanup;
      }
      uint64_t reconf_arg =
          iree_hal_amdxdna_native_buffer_c_device_address(reconf_buffer);
      const iree_device_size_t reconf_offset = 0;
      const iree_device_size_t reconf_length = (iree_device_size_t)seq_bytes;
      for (uint32_t r = 0; r < kernel_params->n_reconfigure_runs; r++) {
        status = iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
            command_buffer, group, kernel_params, /*run_idx=*/2 * i,
            &reconf_arg, &reconf_buffer, &reconf_offset, &reconf_length,
            /*arg_count=*/1, cu_idx, constants, use_native_partial_elf,
            defer_build);
        if (!iree_status_is_ok(status)) goto cleanup;
      }
      status = iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
          command_buffer, group, kernel_params, /*run_idx=*/2 * i + 1,
          binding_addrs, binding_buffers, binding_offsets, binding_lengths,
          bindings.count, cu_idx, constants, use_native_partial_elf,
          defer_build);
      if (!iree_status_is_ok(status)) goto cleanup;
    }
  }

  // Track I/O bindings for residency + final device->host sync at flush.
  // Multiple HAL dispatches in the same command buffer often use the same
  // binding ranges; keep only exact ranges so a 240-dispatch chain does not
  // perform 240 duplicate host invalidations after the parent completes.
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    status = iree_hal_amdxdna_chain_group_append_binding_ref_unique(
        command_buffer->host_allocator, group, bindings.values[j]);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

cleanup:
  iree_allocator_free(command_buffer->host_allocator, binding_lengths);
  iree_allocator_free(command_buffer->host_allocator, binding_offsets);
  iree_allocator_free(command_buffer->host_allocator, binding_buffers);
  iree_allocator_free(command_buffer->host_allocator, binding_addrs);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Build a native ERT_CMD_CHAIN for a contiguous span [begin, end) of a group's
// sub-commands, binding the group's referenced buffers for residency. The
// caller owns submission so native backends can submit all chunks before
// waiting when their DDI supports that shape.
static iree_status_t iree_hal_amdxdna_prepare_chain(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_amdxdna_chain_group_t* group, size_t begin, size_t end,
    iree_hal_amdxdna_native_command_t** out_chain) {
  *out_chain = NULL;
  size_t n = end - begin;
  iree_hal_amdxdna_native_command_t* chain = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_create(
      native_device, IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN,
      &chain));
  iree_hal_amdxdna_native_command_t** commands = NULL;
  iree_status_t status = iree_ok_status();
  if (n != 0) {
    status = iree_allocator_malloc_array(host_allocator, n, sizeof(*commands),
                                         (void**)&commands);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return status;
  }
  for (size_t i = begin; i < end; ++i) {
    commands[i - begin] = group->cmds[i].command;
  }
  status = iree_hal_amdxdna_native_command_c_prepare_chain(chain, commands, n);
  iree_allocator_free(host_allocator, commands);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return status;
  }

  const bool child_bo_table_parent_chain = group->native_partial_elf;
  if (child_bo_table_parent_chain) {
    *out_chain = chain;
    return iree_ok_status();
  }

  // Native address-list chain paths register every BO the firmware dereferences
  // (control code + control-packet data + I/O bindings) as arg BOs on the
  // submitted chain so the driver keeps them resident; the sub-command slots
  // reference them only by address. Module-style partial-ELF chains
  // intentionally skip this superset above: the parent binds child exec BOs
  // only, while each child command BO carries its own BO table.
  const size_t arg_bo_ceiling =
      iree_hal_amdxdna_native_command_c_arg_binding_capacity();
  size_t arg_total = n + group->reconf_buffer_count + group->binding_ref_count;
  if (arg_total > arg_bo_ceiling) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna cmd-chain: %zu arg BOs exceeds native ceiling %zu (chunk "
        "groups or reduce binding count)",
        arg_total, arg_bo_ceiling);
  }
  size_t arg_pos = 0;
  for (size_t i = begin; i < end; i++) {
    status = iree_hal_amdxdna_native_command_c_bind_buffer(
        chain, arg_pos++, group->cmds[i].ctrl_code, 0,
        iree_hal_amdxdna_native_buffer_c_size(group->cmds[i].ctrl_code));
    if (!iree_status_is_ok(status)) break;
  }
  for (iree_host_size_t i = 0; i < group->reconf_buffer_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_amdxdna_native_buffer_t* seq_buffer = group->reconf_buffers[i];
    status = iree_hal_amdxdna_native_command_c_bind_buffer(
        chain, arg_pos++, seq_buffer, 0,
        iree_hal_amdxdna_native_buffer_c_size(seq_buffer));
  }
  for (iree_host_size_t i = 0; i < group->binding_ref_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    const iree_hal_buffer_ref_t* binding_ref = &group->binding_refs[i];
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(binding_ref->buffer));
    status = iree_hal_amdxdna_native_command_c_bind_buffer(
        chain, arg_pos++, native_buffer, 0,
        iree_hal_amdxdna_native_buffer_c_size(native_buffer));
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return status;
  }
  *out_chain = chain;
  return iree_ok_status();
}

static bool iree_hal_amdxdna_chain_group_requires_parent_chain(
    const iree_hal_amdxdna_chain_group_t* group) {
  return iree_hal_amdxdna_chain_group_logical_command_count(group) > 1 ||
         group->reconf_buffer_count != 0;
}

static iree_status_t iree_hal_amdxdna_rebuild_cached_parent_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_command_cache_entry_t* chain_cache,
    uint32_t max_slots) {
  iree_hal_amdxdna_chain_command_cache_entry_clear_chains(
      command_buffer->host_allocator, chain_cache);
  for (size_t begin = 0; begin < chain_cache->group.cmd_count;
       begin += max_slots) {
    size_t end = begin + max_slots;
    if (end > chain_cache->group.cmd_count) end = chain_cache->group.cmd_count;
    iree_hal_amdxdna_native_command_t* chain = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_prepare_chain(
        command_buffer->host_allocator, command_buffer->device->native_device,
        &chain_cache->group, begin, end, &chain));
    iree_status_t append_status =
        iree_hal_amdxdna_chain_command_cache_entry_append_chain(
            command_buffer->host_allocator, chain_cache, chain);
    if (!iree_status_is_ok(append_status)) {
      iree_hal_amdxdna_native_command_c_destroy(chain);
      return append_status;
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_submit_accumulated_single(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group_t* group) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (group->cmd_count != 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "single-dispatch flush expected one command, got "
                            "%zu",
                            group->cmd_count);
  }
  iree_hal_amdxdna_chain_cmd_t* cmd = &group->cmds[0];
  if (!cmd->src_asm_inst || !cmd->src_patches) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "single-dispatch flush is missing recorded control-code descriptors");
  }

  if (!cmd->src_use_native_partial_elf) {
    if (!cmd->built) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_amdxdna_make_npu_cmd(
              command_buffer, cmd->src_cu_idx, cmd->src_asm_inst,
              cmd->src_patches, cmd->binding_device_addrs, cmd->binding_buffers,
              cmd->binding_offsets, cmd->binding_lengths, cmd->binding_count,
              iree_make_const_byte_span(cmd->src_constants,
                                        cmd->src_constant_count),
              cmd->src_use_native_partial_elf, cmd));
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_queue_c_submit_and_wait(
                group->queue, cmd->command, IREE_SV("dispatch")));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  uint32_t* prepared_ctrl_words = NULL;
  iree_hal_amdxdna_native_buffer_t* ctrl_code_buffer = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      command_buffer->host_allocator, cmd->src_asm_inst->count,
      sizeof(*prepared_ctrl_words), (void**)&prepared_ctrl_words);
  if (!iree_status_is_ok(status)) goto cleanup;
  memcpy(prepared_ctrl_words, cmd->src_asm_inst->data,
         cmd->src_asm_inst->count * sizeof(*prepared_ctrl_words));
  status = iree_hal_amdxdna_patch_write32_constants(
      prepared_ctrl_words, cmd->src_asm_inst->count,
      iree_make_const_byte_span(cmd->src_constants, cmd->src_constant_count));
  if (!iree_status_is_ok(status)) goto cleanup;
  if (!iree_hal_amdxdna_apply_patch_table(
          prepared_ctrl_words, cmd->src_asm_inst->count, cmd->src_patches->data,
          cmd->src_patches->count, cmd->binding_device_addrs,
          cmd->binding_count)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna PARTIAL_ELF single dispatch has an invalid host patch table");
    goto cleanup;
  }

  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache =
      iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
  if (!single_command_cache) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "failed to allocate amdxdna single command cache");
    goto cleanup;
  }
  iree_slim_mutex_lock(&single_command_cache->mutex);
  iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = NULL;
  status = iree_hal_amdxdna_find_single_command_cache_entry(
      single_command_cache, group->queue, cmd->src_cu_idx.index,
      prepared_ctrl_words, cmd->src_asm_inst->count, cmd->binding_buffers,
      cmd->binding_device_addrs, cmd->binding_offsets, cmd->binding_lengths,
      cmd->binding_count, &single_cache_entry);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&single_command_cache->mutex);
    goto cleanup;
  }

  iree_hal_amdxdna_native_command_t* submit_command = NULL;
  if (single_cache_entry) {
    submit_command = single_cache_entry->command;
  } else {
    const size_t ctrl_code_size =
        cmd->src_asm_inst->count * sizeof(*prepared_ctrl_words);
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        command_buffer->device->native_device, ctrl_code_size,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &ctrl_code_buffer);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&single_command_cache->mutex);
      goto cleanup;
    }
    void* instr_buffer_ptr = NULL;
    status = iree_hal_amdxdna_native_buffer_c_map(ctrl_code_buffer,
                                                  &instr_buffer_ptr);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&single_command_cache->mutex);
      goto cleanup;
    }
    memcpy(instr_buffer_ptr, prepared_ctrl_words, ctrl_code_size);
    status = iree_hal_amdxdna_native_buffer_c_sync_all(
        ctrl_code_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&single_command_cache->mutex);
      goto cleanup;
    }

    status = iree_hal_amdxdna_native_command_c_create(
        command_buffer->device->native_device,
        IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF,
        &command);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_set_cu_index(command,
                                                              cmd->src_cu_idx);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_control_buffer(
          command, ctrl_code_buffer, ctrl_code_size);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_arg_32(
          command, kAie2ExecBufferKernelOpTxn);
    }
    for (size_t i = 0; i < cmd->binding_count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_amdxdna_native_command_c_bind_buffer(
          command, /*position=*/i + 1, cmd->binding_buffers[i],
          cmd->binding_offsets[i], cmd->binding_lengths[i]);
    }
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&single_command_cache->mutex);
      goto cleanup;
    }
    single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
        single_command_cache, group->queue, cmd->src_cu_idx.index,
        prepared_ctrl_words, cmd->src_asm_inst->count, cmd->binding_buffers,
        cmd->binding_device_addrs, cmd->binding_offsets, cmd->binding_lengths,
        cmd->binding_count, ctrl_code_buffer, command);
    if (!single_cache_entry) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to store amdxdna single command cache "
                                "entry");
      iree_slim_mutex_unlock(&single_command_cache->mutex);
      goto cleanup;
    }
    ctrl_code_buffer = NULL;
    command = NULL;
    submit_command = single_cache_entry->command;
  }
  iree_slim_mutex_unlock(&single_command_cache->mutex);

  status = iree_hal_amdxdna_native_queue_c_submit_and_wait(
      group->queue, submit_command, IREE_SV("dispatch"));

cleanup:
  iree_hal_amdxdna_native_command_c_destroy(command);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrl_code_buffer);
  iree_allocator_free(command_buffer->host_allocator, prepared_ctrl_words);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Flush all accumulated groups. Single-child groups submit directly; groups
// with multiple children become native ERT chains chunked to the backend slot
// limit. Groups are submitted in recorded order so producer/consumer
// dependencies across groups are honored by the device's in-order completion.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_flush_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer) {
  iree_hal_amdxdna_chain_accum_t* accum = &command_buffer->chain_accum;
  if (accum->group_count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  bool has_parent_chain_group = false;
  for (iree_host_size_t i = 0; i < accum->group_count; ++i) {
    if (iree_hal_amdxdna_chain_group_requires_parent_chain(&accum->groups[i])) {
      has_parent_chain_group = true;
      break;
    }
  }

  // Max slots per chain that fit the fixed-size exec buffer (constant per
  // device; computed once and cached). Atomic load with relaxed ordering: a
  // racing first-time probe is idempotent (same value), and the slot count is
  // independent data so we don't need ordering against any other state. Use
  // acquire on the success path / release on the store so a thread observing
  // the cached value also observes the probe's published writes.
  uint32_t max_slots = UINT32_MAX;
  if (has_parent_chain_group) {
    max_slots = iree_atomic_load(&command_buffer->device->chain_max_slots,
                                 iree_memory_order_acquire);
    if (max_slots == 0) {
      max_slots = command_buffer->device->native_caps.max_command_chain_slots;
      if (max_slots == 0) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_native_device_c_query_chain_max_slots(
                    command_buffer->device->native_device, &max_slots));
      }
      if (max_slots == 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "amdxdna command-chain backend reported zero max slots");
      }
      iree_atomic_store(&command_buffer->device->chain_max_slots, max_slots,
                        iree_memory_order_release);
    }
  }

  // Submit each accumulated group either as one direct native command or as
  // native ERT chains chunked into max_slots-sized pieces.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t group_index = 0;
       group_index < accum->group_count && iree_status_is_ok(status);
       ++group_index) {
    iree_hal_amdxdna_chain_group_t* group = &accum->groups[group_index];
    const bool submit_as_chain =
        iree_hal_amdxdna_chain_group_requires_parent_chain(group);
    if (!submit_as_chain) {
      status = iree_hal_amdxdna_direct_command_buffer_submit_accumulated_single(
          command_buffer, group);
    } else {
      iree_hal_amdxdna_chain_command_cache_entry_t* chain_cache = NULL;
      if (group->native_partial_elf && group->reconf_buffer_count == 0) {
        iree_hal_amdxdna_device_chain_command_cache_t* device_chain_cache =
            iree_hal_amdxdna_get_chain_command_cache(command_buffer->device);
        if (!device_chain_cache) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "failed to allocate amdxdna chain command cache");
          break;
        }
        iree_slim_mutex_lock(&device_chain_cache->mutex);
        bool exact_cache_hit = false;
        bool device_cache_hit = false;
        // Deferred-build fast path: reuse an already-built cached chain when
        // the descriptor inputs (control-code template + constants + bindings)
        // match exactly, without building this group's children at all.
        for (iree_host_size_t i = 0; i < device_chain_cache->entry_count; ++i) {
          iree_hal_amdxdna_chain_command_cache_entry_t* entry =
              &device_chain_cache->entries[i];
          if (iree_hal_amdxdna_chain_command_cache_descriptor_matches(
                  entry, group, max_slots)) {
            chain_cache = entry;
            exact_cache_hit = true;
            break;
          }
        }
        // Not an exact hit: realize any deferred children now so the
        // ctrl_words-based device/shape/miss logic below can match, update, or
        // cache them.
        if (!chain_cache) {
          status = iree_hal_amdxdna_expand_repeated_chain_descriptors(
              command_buffer->host_allocator, group);
        }
        if (!chain_cache && iree_status_is_ok(status)) {
          for (iree_host_size_t i = 0;
               i < group->cmd_count && iree_status_is_ok(status); ++i) {
            iree_hal_amdxdna_chain_cmd_t* cmd = &group->cmds[i];
            if (cmd->built) continue;
            status = iree_hal_amdxdna_make_npu_cmd(
                command_buffer, cmd->src_cu_idx, cmd->src_asm_inst,
                cmd->src_patches, cmd->binding_device_addrs,
                cmd->binding_buffers, cmd->binding_offsets,
                cmd->binding_lengths, cmd->binding_count,
                iree_make_const_byte_span(cmd->src_constants,
                                          cmd->src_constant_count),
                cmd->src_use_native_partial_elf, cmd);
          }
        }
        if (chain_cache) {
          chain_cache->last_use = ++device_chain_cache->use_clock;
        } else if (iree_status_is_ok(status)) {
          for (iree_host_size_t i = 0; i < device_chain_cache->entry_count;
               ++i) {
            iree_hal_amdxdna_chain_command_cache_entry_t* entry =
                &device_chain_cache->entries[i];
            if (iree_hal_amdxdna_chain_command_cache_device_matches(
                    entry, group, max_slots)) {
              chain_cache = entry;
              break;
            }
          }
          if (chain_cache) {
            device_cache_hit = true;
            chain_cache->last_use = ++device_chain_cache->use_clock;
          }
        }
        if (iree_status_is_ok(status) && !chain_cache &&
            device_chain_cache->entry_count >=
                kAmdxdnaChainCommandCacheCapacity) {
          for (iree_host_size_t i = 0; i < device_chain_cache->entry_count;
               ++i) {
            iree_hal_amdxdna_chain_command_cache_entry_t* entry =
                &device_chain_cache->entries[i];
            if (iree_hal_amdxdna_chain_command_cache_shape_matches(entry, group,
                                                                   max_slots)) {
              chain_cache = entry;
              break;
            }
          }
        }
        if (chain_cache && !exact_cache_hit && !device_cache_hit) {
          chain_cache->last_use = ++device_chain_cache->use_clock;
          bool packet_changed = false;
          for (size_t i = 0; i < group->cmd_count && iree_status_is_ok(status);
               ++i) {
            bool cmd_packet_changed = false;
            status = iree_hal_amdxdna_update_cached_chain_cmd(
                &chain_cache->group.cmds[i], &group->cmds[i],
                &cmd_packet_changed, NULL, NULL, NULL);
            if (cmd_packet_changed) {
              packet_changed = true;
            }
          }
          if (packet_changed) {
            for (iree_host_size_t i = 0;
                 i < chain_cache->chain_count && iree_status_is_ok(status);
                 ++i) {
              if (!iree_status_is_ok(status)) break;
              status = iree_hal_amdxdna_native_command_c_mark_chain_code_dirty(
                  chain_cache->chains[i]);
            }
          }
        } else if (iree_status_is_ok(status) && !chain_cache) {
          chain_cache = iree_hal_amdxdna_chain_command_cache_allocate_entry(
              device_chain_cache);
          chain_cache->group.context =
              iree_hal_amdxdna_native_context_ref_retain(group->context);
          chain_cache->group.queue = group->queue;
          chain_cache->group.native_partial_elf = group->native_partial_elf;
          status = iree_hal_amdxdna_chain_group_take_cmds(
              command_buffer->host_allocator, &chain_cache->group, group);
          chain_cache->max_slots = max_slots;
          chain_cache->last_use = ++device_chain_cache->use_clock;
          if (iree_status_is_ok(status)) {
            status = iree_hal_amdxdna_rebuild_cached_parent_chains(
                command_buffer, chain_cache, max_slots);
          }
        }
        if (iree_status_is_ok(status) && chain_cache->chain_count != 0) {
          status = iree_hal_amdxdna_native_queue_c_submit_all_and_wait(
              group->queue, chain_cache->chains, chain_cache->chain_count,
              IREE_SV("ERT_CMD_CHAIN"));
        }
        iree_slim_mutex_unlock(&device_chain_cache->mutex);
      } else {
        iree_hal_amdxdna_native_command_t** chains = NULL;
        size_t chain_count = 0;
        size_t chain_capacity =
            (group->cmd_count + (size_t)max_slots - 1) / (size_t)max_slots;
        if (chain_capacity != 0) {
          status = iree_allocator_malloc_array(command_buffer->host_allocator,
                                               chain_capacity, sizeof(*chains),
                                               (void**)&chains);
        }
        for (size_t begin = 0;
             begin < group->cmd_count && iree_status_is_ok(status);
             begin += max_slots) {
          size_t end = begin + max_slots;
          if (end > group->cmd_count) end = group->cmd_count;
          iree_hal_amdxdna_native_command_t* chain = NULL;
          status = iree_hal_amdxdna_prepare_chain(
              command_buffer->host_allocator,
              command_buffer->device->native_device, group, begin, end, &chain);
          if (iree_status_is_ok(status)) chains[chain_count++] = chain;
        }
        if (iree_status_is_ok(status) && chain_count != 0) {
          status = iree_hal_amdxdna_native_queue_c_submit_all_and_wait(
              group->queue, chains, chain_count, IREE_SV("ERT_CMD_CHAIN"));
        }
        for (size_t i = 0; i < chain_count; ++i) {
          iree_hal_amdxdna_native_command_c_destroy(chains[i]);
        }
        iree_allocator_free(command_buffer->host_allocator, chains);
      }
    }
    if (!iree_status_is_ok(status)) break;
    // Parent chains do not go through the native submit path's normal binding
    // sync model, so sync this group's I/O bindings back to host once the
    // chains complete. The direct single path mirrors normal_run() and relies
    // on the native submit's binding sync behavior.
    if (submit_as_chain) {
      for (iree_host_size_t i = 0; i < group->binding_ref_count; ++i) {
        const iree_hal_buffer_ref_t* binding_ref = &group->binding_refs[i];
        status = iree_hal_amdxdna_buffer_invalidate_range(
            binding_ref->buffer, binding_ref->offset, binding_ref->length);
        if (!iree_status_is_ok(status)) break;
      }
    }
    if (!iree_status_is_ok(status)) break;
  }
  // Drop the accumulator unconditionally. On the OK path this is the normal
  // post-flush reset; on the error path it makes sure the remaining
  // unsubmitted groups (their control-code BOs, sub-command BOs, reconf BOs)
  // release HERE rather than during the command-buffer destructor's unwind
  // while it's already propagating the error up. No leak either way: the
  // destructor would run them, but this keeps the failure site self-contained
  // (anything still pending at the error point is gone) and avoids running BO
  // destructors mid error-unwind, which is hard to read in a crash trace.
  iree_hal_amdxdna_chain_accum_clear(command_buffer->host_allocator, accum);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_normal_run(
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patch_table,
    iree_const_byte_span_t constants, bool use_single_partial_elf) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  uint64_t* binding_addrs = NULL;
  iree_hal_amdxdna_native_buffer_t** binding_buffers = NULL;
  iree_device_size_t* binding_offsets = NULL;
  iree_device_size_t* binding_lengths = NULL;
  uint32_t* prepared_ctrl_words = NULL;
  iree_hal_amdxdna_native_buffer_t* ctrl_code_buffer = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;
  iree_hal_amdxdna_native_command_t* submit_command = NULL;
  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache = NULL;
  iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = NULL;
  bool single_cache_locked = false;

  if (use_single_partial_elf) {
    if (bindings.count != 0) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, bindings.count,
          sizeof(*binding_addrs), (void**)&binding_addrs);
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            command_buffer->host_allocator, bindings.count,
            sizeof(*binding_buffers), (void**)&binding_buffers);
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            command_buffer->host_allocator, bindings.count,
            sizeof(*binding_offsets), (void**)&binding_offsets);
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            command_buffer->host_allocator, bindings.count,
            sizeof(*binding_lengths), (void**)&binding_lengths);
      }
      if (!iree_status_is_ok(status)) goto cleanup;
    }
    for (iree_host_size_t j = 0; j < bindings.count; ++j) {
      iree_hal_amdxdna_native_buffer_t* native_buffer =
          iree_hal_amdxdna_buffer_handle(
              iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
      status = iree_hal_amdxdna_native_buffer_c_ensure_allocated(native_buffer);
      if (!iree_status_is_ok(status)) goto cleanup;
      const iree_device_size_t native_offset =
          iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
          bindings.values[j].offset;
      binding_addrs[j] =
          iree_hal_amdxdna_native_buffer_c_device_address(native_buffer) +
          native_offset;
      binding_buffers[j] = native_buffer;
      binding_offsets[j] = native_offset;
      binding_lengths[j] = bindings.values[j].length;
    }
  }

  // Production policy: reuse a prepared single-dispatch native command across
  // queue_execute calls (keyed by the dispatch signature in the device single-
  // command cache) instead of rebuilding it each time. Always on; kept as a
  // named flag until the prepared-command model in the native-DDI follow-ups
  // replaces the device-global caches.
  const bool use_single_command_cache = true;
  if (use_single_partial_elf) {
    status = iree_allocator_malloc_array(
        command_buffer->host_allocator, asm_inst->count,
        sizeof(*prepared_ctrl_words), (void**)&prepared_ctrl_words);
    if (!iree_status_is_ok(status)) goto cleanup;
    memcpy(prepared_ctrl_words, asm_inst->data,
           asm_inst->count * sizeof(*prepared_ctrl_words));
    status = iree_hal_amdxdna_patch_write32_constants(
        prepared_ctrl_words, asm_inst->count, constants);
    if (!iree_status_is_ok(status)) goto cleanup;
    if (!iree_hal_amdxdna_apply_patch_table(
            prepared_ctrl_words, asm_inst->count, patch_table->data,
            patch_table->count, binding_addrs, bindings.count)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna PARTIAL_ELF single dispatch has an invalid host "
          "patch table");
      goto cleanup;
    }
    if (use_single_command_cache) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to allocate amdxdna single command cache");
        goto cleanup;
      }
      iree_slim_mutex_lock(&single_command_cache->mutex);
      single_cache_locked = true;
      status = iree_hal_amdxdna_find_single_command_cache_entry(
          single_command_cache, queue, cu_idx.index, prepared_ctrl_words,
          asm_inst->count, binding_buffers, binding_addrs, binding_offsets,
          binding_lengths, bindings.count, &single_cache_entry);
      if (!iree_status_is_ok(status)) goto cleanup;
      if (single_cache_entry) {
        submit_command = single_cache_entry->command;
      } else {
        iree_slim_mutex_unlock(&single_command_cache->mutex);
        single_cache_locked = false;
      }
    }
  }

  // Allocate a buffer object to hold the control code (`asm_inst`).
  size_t ctrl_code_size =
      (use_single_partial_elf ? asm_inst->count : asm_inst->count) *
      sizeof(uint32_t);
  if (!submit_command) {
    const bool uses_native_instruction_buffer =
        use_single_partial_elf ||
        (command_buffer->device->native_caps.dispatch_models &
         IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        command_buffer->device->native_device, ctrl_code_size,
        uses_native_instruction_buffer
            ? IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION
            : IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE,
        &ctrl_code_buffer);
    if (!iree_status_is_ok(status)) goto cleanup;
  }
  void* instr_buffer_ptr = NULL;
  uint32_t* instr_buffer = NULL;
  if (!submit_command) {
    status = iree_hal_amdxdna_native_buffer_c_map(ctrl_code_buffer,
                                                  &instr_buffer_ptr);
    if (!iree_status_is_ok(status)) goto cleanup;
    instr_buffer = (uint32_t*)instr_buffer_ptr;
    if (use_single_partial_elf) {
      memcpy(instr_buffer, prepared_ctrl_words, ctrl_code_size);
    } else {
      memcpy(instr_buffer, asm_inst->data, ctrl_code_size);
      status = iree_hal_amdxdna_patch_write32_constants(
          instr_buffer, asm_inst->count, constants);
      if (!iree_status_is_ok(status)) goto cleanup;
    }
    status = iree_hal_amdxdna_native_buffer_c_sync_all(
        ctrl_code_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  const iree_hal_amdxdna_native_c_command_opcode_t command_opcode =
      use_single_partial_elf
          ? IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF
          : command_buffer->device->native_caps.default_dispatch_opcode;
  if (!submit_command) {
    status = iree_hal_amdxdna_native_command_c_create(
        command_buffer->device->native_device, command_opcode, &command);
    if (!iree_status_is_ok(status)) goto cleanup;
    // Add the kernel arguments.
    status = iree_hal_amdxdna_native_command_c_set_cu_index(command, cu_idx);
    if (!iree_status_is_ok(status)) goto cleanup;
    if (use_single_partial_elf) {
      status = iree_hal_amdxdna_native_command_c_add_control_buffer(
          command, ctrl_code_buffer, ctrl_code_size);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_arg_32(
            command, kAie2ExecBufferKernelOpTxn);
      }
    } else if ((command_buffer->device->native_caps.dispatch_models &
                IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0) {
      status = iree_hal_amdxdna_native_command_c_add_control_buffer(
          command, ctrl_code_buffer, ctrl_code_size);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_arg_32(
            command, kAie2ExecBufferKernelOpTxn);
      }
    } else {
      status = iree_hal_amdxdna_native_command_c_add_arg_64(
          command, kAmdxdnaControlCodeOpcode);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_buffer_arg(
            command, ctrl_code_buffer);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_arg_32(
            command, (uint32_t)asm_inst->count);
      }
    }
    if (!iree_status_is_ok(status)) goto cleanup;
  }

  if (!submit_command) {
    if (use_single_partial_elf) {
      for (iree_host_size_t j = 0; j < bindings.count; ++j) {
        status = iree_hal_amdxdna_native_command_c_bind_buffer(
            command, /*position=*/j + 1, binding_buffers[j], binding_offsets[j],
            binding_lengths[j]);
        if (!iree_status_is_ok(status)) goto cleanup;
      }
    } else {
      for (iree_host_size_t j = 0; j < bindings.count; ++j) {
        iree_hal_amdxdna_native_buffer_t* native_buffer =
            iree_hal_amdxdna_buffer_handle(
                iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
        // Propagate per-binding byte_offset (both the buffer's own subview
        // offset within its allocated root, and the binding-level offset) into
        // the device-side address. Without this, two bindings on the same root
        // BO at different offsets collapse to the same physical address,
        // causing the next dispatch to read/write the wrong slot.
        uint64_t buffer_byte_off =
            (uint64_t)iree_hal_buffer_byte_offset(bindings.values[j].buffer);
        uint64_t binding_off = (uint64_t)bindings.values[j].offset;
        iree_device_size_t native_offset = buffer_byte_off + binding_off;
        status = iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
            command, native_buffer, native_offset);
        if (!iree_status_is_ok(status)) goto cleanup;
      }
    }
  }

  if (use_single_partial_elf && use_single_command_cache && !submit_command) {
    if (!single_command_cache) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to allocate amdxdna single command cache");
        goto cleanup;
      }
    }
    if (!single_cache_locked) {
      iree_slim_mutex_lock(&single_command_cache->mutex);
      single_cache_locked = true;
    }
    // Another queue worker may have populated the entry while this thread was
    // building the native command. Recheck under the cache lock before storing.
    status = iree_hal_amdxdna_find_single_command_cache_entry(
        single_command_cache, queue, cu_idx.index, prepared_ctrl_words,
        asm_inst->count, binding_buffers, binding_addrs, binding_offsets,
        binding_lengths, bindings.count, &single_cache_entry);
    if (!iree_status_is_ok(status)) goto cleanup;
    if (!single_cache_entry) {
      single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
          single_command_cache, queue, cu_idx.index, prepared_ctrl_words,
          asm_inst->count, binding_buffers, binding_addrs, binding_offsets,
          binding_lengths, bindings.count, ctrl_code_buffer, command);
      if (!single_cache_entry) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to store amdxdna single command cache "
                             "entry");
        goto cleanup;
      }
      ctrl_code_buffer = NULL;
      command = NULL;
    }
    submit_command = single_cache_entry->command;
  }

  if (!submit_command) {
    submit_command = command;
  }

  status = iree_hal_amdxdna_native_queue_c_submit_and_wait(
      queue, submit_command, IREE_SV("dispatch"));
  if (!iree_status_is_ok(status)) goto cleanup;
  // Sync the bindings back to the host.
  if (!use_single_partial_elf &&
      command_buffer->device->native_caps.buffer_sync_model !=
          IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_SUBMIT_SYNCS_BINDINGS) {
    for (iree_host_size_t j = 0; j < bindings.count; ++j) {
      status = iree_hal_amdxdna_buffer_invalidate_range(
          bindings.values[j].buffer, bindings.values[j].offset,
          bindings.values[j].length);
      if (!iree_status_is_ok(status)) goto cleanup;
    }
  }

cleanup:
  if (single_cache_locked) iree_slim_mutex_unlock(&single_command_cache->mutex);
  iree_hal_amdxdna_native_command_c_destroy(command);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrl_code_buffer);
  iree_allocator_free(command_buffer->host_allocator, prepared_ctrl_words);
  iree_allocator_free(command_buffer->host_allocator, binding_lengths);
  iree_allocator_free(command_buffer->host_allocator, binding_offsets);
  iree_allocator_free(command_buffer->host_allocator, binding_buffers);
  iree_allocator_free(command_buffer->host_allocator, binding_addrs);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reconfigure(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx, uint32_t n_reconfigure_runs,
    const iree_hal_amdxdna_u32_list_t* ctrlpkt_inst,
    const iree_hal_amdxdna_u32_list_t* ctrlpkt_seq,
    iree_const_byte_span_t constants) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  iree_hal_amdxdna_native_buffer_t* ctrlpkt_inst_buffer = NULL;
  iree_hal_amdxdna_native_buffer_t* ctrlpkt_seq_buffer = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;

  // Allocate a buffer object to hold the control packet instructions.
  size_t ctrlpkt_inst_size = ctrlpkt_inst->count * sizeof(uint32_t);
  const bool uses_native_instruction_buffer =
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0;
  status = iree_hal_amdxdna_native_device_c_alloc_buffer(
      command_buffer->device->native_device, ctrlpkt_inst_size,
      uses_native_instruction_buffer
          ? IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION
          : IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE,
      &ctrlpkt_inst_buffer);
  if (!iree_status_is_ok(status)) goto cleanup;
  void* ctrlpkt_inst_ptr = NULL;
  status = iree_hal_amdxdna_native_buffer_c_map(ctrlpkt_inst_buffer,
                                                &ctrlpkt_inst_ptr);
  if (!iree_status_is_ok(status)) goto cleanup;
  uint32_t* ctrlpkt_inst_words = (uint32_t*)ctrlpkt_inst_ptr;
  memcpy(ctrlpkt_inst_words, ctrlpkt_inst->data, ctrlpkt_inst_size);
  status = iree_hal_amdxdna_patch_write32_constants(
      ctrlpkt_inst_words, ctrlpkt_inst->count, constants);
  if (!iree_status_is_ok(status)) goto cleanup;
  status = iree_hal_amdxdna_native_buffer_c_sync_all(
      ctrlpkt_inst_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
  if (!iree_status_is_ok(status)) goto cleanup;

  // Allocate a buffer object to hold the control packet sequence (content).
  size_t ctrlpkt_seq_size = ctrlpkt_seq->count * sizeof(uint32_t);
  status = iree_hal_amdxdna_native_device_c_alloc_buffer(
      command_buffer->device->native_device, ctrlpkt_seq_size,
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY, &ctrlpkt_seq_buffer);
  if (!iree_status_is_ok(status)) goto cleanup;
  void* ctrlpkt_seq_ptr = NULL;
  status = iree_hal_amdxdna_native_buffer_c_map(ctrlpkt_seq_buffer,
                                                &ctrlpkt_seq_ptr);
  if (!iree_status_is_ok(status)) goto cleanup;
  memcpy(ctrlpkt_seq_ptr, ctrlpkt_seq->data, ctrlpkt_seq_size);
  status = iree_hal_amdxdna_native_buffer_c_sync_all(
      ctrlpkt_seq_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_hal_amdxdna_native_command_c_create(
      command_buffer->device->native_device,
      command_buffer->device->native_caps.default_dispatch_opcode, &command);
  if (!iree_status_is_ok(status)) goto cleanup;

  // Add the kernel arguments.
  status = iree_hal_amdxdna_native_command_c_set_cu_index(command, cu_idx);
  if (!iree_status_is_ok(status)) goto cleanup;
  if ((command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0) {
    status = iree_hal_amdxdna_native_command_c_add_control_buffer(
        command, ctrlpkt_inst_buffer, ctrlpkt_inst_size);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = iree_hal_amdxdna_native_command_c_add_arg_32(
        command, kAie2ExecBufferKernelOpTxn);
    if (!iree_status_is_ok(status)) goto cleanup;
  } else {
    status = iree_hal_amdxdna_native_command_c_add_arg_64(
        command, kAmdxdnaControlCodeOpcode);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = iree_hal_amdxdna_native_command_c_add_buffer_arg(
        command, ctrlpkt_inst_buffer);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = iree_hal_amdxdna_native_command_c_add_arg_32(
        command, (uint32_t)ctrlpkt_inst->count);
    if (!iree_status_is_ok(status)) goto cleanup;
  }
  status = iree_hal_amdxdna_native_command_c_add_buffer_arg(command,
                                                            ctrlpkt_seq_buffer);
  if (!iree_status_is_ok(status)) goto cleanup;

  // Execute the reconfiguration for `n_reconfigure_runs` times.
  for (uint32_t i = 0; i < n_reconfigure_runs; ++i) {
    status = iree_hal_amdxdna_native_queue_c_submit_and_wait(
        queue, command, IREE_SV("control-packet reconfiguration"));
    if (!iree_status_is_ok(status)) goto cleanup;
  }

cleanup:
  iree_hal_amdxdna_native_command_c_destroy(command);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrlpkt_seq_buffer);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrlpkt_inst_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  iree_hal_amdxdna_native_context_ref_t* context_ref = NULL;

  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  // Look up kernel parameters used for side-channeling additional launch
  // information from the compiler. Bound by reference: the executable owns
  // the params for its lifetime and we only read them here, so we avoid
  // copying the struct (which holds the PDI bytes plus several runlists) on
  // every dispatch.
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "entry point function %" PRIu64
                              " out of range; executable only contains %" PRIhsz
                              " entry points",
                              function.value, executable->entry_point_count);
    goto cleanup;
  }
  const uint32_t entry_point = iree_hal_executable_function_index(function);
  if (entry_point >= executable->entry_point_count) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "entry point ordinal %u out of range; executable "
                              "only contains %" PRIhsz " entry points",
                              entry_point, executable->entry_point_count);
    goto cleanup;
  }
  iree_hal_amdxdna_kernel_params_t* kernel_params =
      &executable->entry_points[entry_point];
  status = iree_hal_amdxdna_validate_live_dispatch_bindings(bindings);
  if (!iree_status_is_ok(status)) goto cleanup;

  status = iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                        &executable);
  if (!iree_status_is_ok(status)) goto cleanup;

  size_t num_reconfigurations = kernel_params->reconf_data_runlist_count;
  iree_const_byte_span_t pdi_span = iree_make_const_byte_span(
      kernel_params->pdi.data, kernel_params->pdi.count);
  iree_const_byte_span_t xclbin_span = iree_make_const_byte_span(
      kernel_params->xclbin.data, kernel_params->xclbin.count);
  iree_string_view_t kernel_name = kernel_params->kernel_name;
  const iree_hal_amdxdna_u32_list_t* single_patch_table =
      kernel_params->patch_runlist_count == 0
          ? NULL
          : &kernel_params->patch_runlist[0];
  const bool use_native_partial_elf_context =
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF) != 0 &&
      num_reconfigurations == 0 && xclbin_span.data_length > 0 &&
      iree_hal_amdxdna_patch_table_is_valid(single_patch_table);
  iree_hal_amdxdna_native_c_cu_index_t cu_idx;
  memset(&cu_idx, 0, sizeof(cu_idx));
  if (num_reconfigurations == 0) {
    if (use_native_partial_elf_context) {
      // Memoize the resolved context + CU on the entry point. Otherwise every
      // dispatch re-enters get_or_create_context, which copies and FNV-hashes
      // the whole PDI under a lock; that dominates a 240-dispatch chain. Repeat
      // dispatches reuse the cached context and skip the CU open entirely.
      iree_slim_mutex_lock(&executable->context_mutex);
      if (kernel_params->cached_context_valid) {
        context_ref = iree_hal_amdxdna_native_context_ref_retain(
            kernel_params->cached_context);
        cu_idx = kernel_params->cached_cu_index;
      } else {
        iree_hal_amdxdna_native_context_ref_t* raw_context_ref = NULL;
        status = iree_hal_amdxdna_device_get_or_create_context(
            command_buffer->device, pdi_span, xclbin_span, kernel_name,
            &raw_context_ref);
        if (iree_status_is_ok(status)) {
          context_ref = raw_context_ref;
          status = iree_hal_amdxdna_native_context_ref_open_cu(
              context_ref, kernel_name, &cu_idx);
        }
        if (iree_status_is_ok(status)) {
          kernel_params->cached_context =
              iree_hal_amdxdna_native_context_ref_retain(context_ref);
          kernel_params->cached_cu_index = cu_idx;
          kernel_params->cached_context_valid = true;
        }
      }
      iree_slim_mutex_unlock(&executable->context_mutex);
      if (!iree_status_is_ok(status)) goto cleanup;
    } else {
      iree_hal_amdxdna_native_c_context_image_t context_image;
      memset(&context_image, 0, sizeof(context_image));
      context_image.pdi = pdi_span;
      context_image.xclbin = xclbin_span;
      context_image.kernel_name = kernel_name;
      context_image.type =
          (xclbin_span.data_length != 0 &&
           (command_buffer->device->native_caps.context_image_models &
            IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_XCLBIN) != 0)
              ? IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN
              : IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI;
      status = iree_hal_amdxdna_native_device_c_create_context_ref(
          command_buffer->device->native_device, &context_image, &context_ref);
      if (!iree_status_is_ok(status)) goto cleanup;
      status = iree_hal_amdxdna_native_context_ref_open_cu(
          context_ref, kernel_name, &cu_idx);
      if (!iree_status_is_ok(status)) goto cleanup;
    }
  } else if (kernel_params->pdi.count != 0 ||
             kernel_params->xclbin.count != 0) {
    status = iree_hal_amdxdna_device_get_or_create_context(
        command_buffer->device, pdi_span, xclbin_span, kernel_name,
        &context_ref);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = iree_hal_amdxdna_native_context_ref_open_cu(context_ref,
                                                         kernel_name, &cu_idx);
    if (!iree_status_is_ok(status)) goto cleanup;
    {
      iree_slim_mutex_lock(&executable->context_mutex);
      iree_hal_amdxdna_native_context_ref_release(executable->context);
      executable->context =
          iree_hal_amdxdna_native_context_ref_retain(context_ref);
      executable->context_cu_index = cu_idx;
      executable->context_cu_index_valid = true;
      iree_slim_mutex_unlock(&executable->context_mutex);
    }
  } else {
    iree_slim_mutex_lock(&executable->context_mutex);
    if (executable->context_cu_index_valid) {
      context_ref =
          iree_hal_amdxdna_native_context_ref_retain(executable->context);
      cu_idx = executable->context_cu_index;
    }
    iree_slim_mutex_unlock(&executable->context_mutex);
  }
  if (!context_ref) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna: control-packet dispatch with no context image ran before "
        "its PDI/xclbin-carrying entry point loaded the array");
    goto cleanup;
  }
  iree_hal_amdxdna_native_queue_t* queue =
      iree_hal_amdxdna_native_context_ref_queue(context_ref);

  const bool has_host_patch_table = kernel_params->patch_runlist_count ==
                                    kernel_params->asm_inst_runlist_count;
  const bool multi_control_code_or_pdi =
      kernel_params->n_pdi_loads > 1 ||
      kernel_params->asm_inst_runlist_count > 1 || num_reconfigurations != 0;
  const bool use_deferred_submit_policy =
      has_host_patch_table &&
      (use_native_partial_elf_context || multi_control_code_or_pdi);
  if (use_deferred_submit_policy) {
    // Accumulate this dispatch's commands; end() chooses the native shape from
    // the recorded stream: one child submits as a direct single dispatch, while
    // multiple children or multi-control-code artifacts become ERT_CMD_CHAINs.
    status = iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
        bindings, command_buffer, context_ref, queue, cu_idx, kernel_params,
        constants, use_native_partial_elf_context);
    goto cleanup;
  }

  if (num_reconfigurations == 0) {
    // Normal kernel dispatch.
    status = iree_hal_amdxdna_direct_command_buffer_normal_run(
        bindings, command_buffer, queue, cu_idx,
        &kernel_params->asm_inst_runlist[0], single_patch_table, constants,
        use_native_partial_elf_context);
    if (!iree_status_is_ok(status)) goto cleanup;
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Reconfigure the device.
      status = iree_hal_amdxdna_direct_command_buffer_reconfigure(
          command_buffer, queue, cu_idx, kernel_params->n_reconfigure_runs,
          &kernel_params->asm_inst_runlist[2 * i],
          &kernel_params->reconf_data_runlist[i], constants);
      if (!iree_status_is_ok(status)) goto cleanup;
      // Dispatch the new kernel.
      const size_t run_idx = 2 * i + 1;
      const iree_hal_amdxdna_u32_list_t* patch_table =
          run_idx < kernel_params->patch_runlist_count
              ? &kernel_params->patch_runlist[run_idx]
              : NULL;
      status = iree_hal_amdxdna_direct_command_buffer_normal_run(
          bindings, command_buffer, queue, cu_idx,
          &kernel_params->asm_inst_runlist[run_idx], patch_table, constants,
          /*use_single_partial_elf=*/false);
      if (!iree_status_is_ok(status)) goto cleanup;
    }
  }

cleanup:
  iree_hal_amdxdna_native_context_ref_release(context_ref);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Flush deferred dispatches once the whole command buffer has been recorded.
// No-op when every dispatch used the immediate legacy path.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_status_t status =
      iree_hal_amdxdna_direct_command_buffer_flush_chains(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDXDNA collectives are not implemented");
}

const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable = {
        iree_hal_amdxdna_direct_command_buffer_destroy,
        iree_hal_amdxdna_direct_command_buffer_begin,
        iree_hal_amdxdna_direct_command_buffer_end,
        iree_hal_amdxdna_direct_command_buffer_begin_debug_group,
        iree_hal_amdxdna_direct_command_buffer_end_debug_group,
        iree_hal_amdxdna_direct_command_buffer_execution_barrier,
        iree_hal_amdxdna_direct_command_buffer_signal_event,
        iree_hal_amdxdna_direct_command_buffer_reset_event,
        iree_hal_amdxdna_direct_command_buffer_wait_events,
        iree_hal_amdxdna_direct_command_buffer_advise_buffer,
        iree_hal_amdxdna_direct_command_buffer_fill_buffer,
        iree_hal_amdxdna_direct_command_buffer_update_buffer,
        iree_hal_amdxdna_direct_command_buffer_copy_buffer,
        iree_hal_amdxdna_direct_command_buffer_collective,
        iree_hal_amdxdna_direct_command_buffer_dispatch,
};
