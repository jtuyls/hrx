// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/device_internal.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/utils/resource_set.h"

static constexpr uint64_t kAmdxdnaControlCodeOpcode = 3u;

static bool iree_hal_amdxdna_patch_table_is_valid(
    const iree_hal_amdxdna_u32_list_t* patch_table) {
  return patch_table && patch_table->count != 0 && (patch_table->count % 3 == 0);
}

struct iree_hal_amdxdna_direct_command_buffer {
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
  iree_hal_amdxdna_chain_accum chain_accum;

};

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

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable;

static iree_hal_amdxdna_native_command_opcode_t
iree_hal_amdxdna_native_command_opcode_from_c(
    iree_hal_amdxdna_native_c_command_opcode_t opcode) {
  switch (opcode) {
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_CU:
      return iree_hal_amdxdna_native_command_opcode_t::start_cu;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU:
      return iree_hal_amdxdna_native_command_opcode_t::start_npu;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF:
      return iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf;
    case IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN:
      return iree_hal_amdxdna_native_command_opcode_t::command_chain;
  }
  return iree_hal_amdxdna_native_command_opcode_t::start_cu;
}

static iree_hal_amdxdna_native_cu_index_t
iree_hal_amdxdna_native_cu_index_from_c(
    iree_hal_amdxdna_native_c_cu_index_t cu_index) {
  iree_hal_amdxdna_native_cu_index_t native_cu_index;
  native_cu_index.index = cu_index.index;
  return native_cu_index;
}

struct AmdxdnaSlimMutexLock {
  explicit AmdxdnaSlimMutexLock(iree_slim_mutex_t* mutex) : mutex(mutex) {
    iree_slim_mutex_lock(mutex);
  }
  ~AmdxdnaSlimMutexLock() { iree_slim_mutex_unlock(mutex); }
  iree_slim_mutex_t* mutex = nullptr;
};

struct AmdxdnaSlimMutexUniqueLock {
  AmdxdnaSlimMutexUniqueLock() = default;
  explicit AmdxdnaSlimMutexUniqueLock(iree_slim_mutex_t* mutex)
      : mutex(mutex), owns(true) {
    iree_slim_mutex_lock(mutex);
  }
  ~AmdxdnaSlimMutexUniqueLock() {
    if (owns) iree_slim_mutex_unlock(mutex);
  }
  AmdxdnaSlimMutexUniqueLock(const AmdxdnaSlimMutexUniqueLock&) = delete;
  AmdxdnaSlimMutexUniqueLock& operator=(
      const AmdxdnaSlimMutexUniqueLock&) = delete;
  AmdxdnaSlimMutexUniqueLock(AmdxdnaSlimMutexUniqueLock&& other) noexcept
      : mutex(other.mutex), owns(other.owns) {
    other.mutex = nullptr;
    other.owns = false;
  }
  AmdxdnaSlimMutexUniqueLock& operator=(
      AmdxdnaSlimMutexUniqueLock&& other) noexcept {
    if (this == &other) return *this;
    if (owns) iree_slim_mutex_unlock(mutex);
    mutex = other.mutex;
    owns = other.owns;
    other.mutex = nullptr;
    other.owns = false;
    return *this;
  }
  void lock(iree_slim_mutex_t* new_mutex) {
    if (owns) iree_slim_mutex_unlock(mutex);
    mutex = new_mutex;
    owns = true;
    iree_slim_mutex_lock(mutex);
  }
  void unlock() {
    if (!owns) return;
    iree_slim_mutex_unlock(mutex);
    owns = false;
  }
  bool owns_lock() const { return owns; }
  iree_slim_mutex_t* mutex = nullptr;
  bool owns = false;
};
}  // namespace

iree_status_t iree_hal_amdxdna_direct_command_buffer_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = nullptr;
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

  iree_hal_amdxdna_direct_command_buffer* command_buffer = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            reinterpret_cast<void**>(&command_buffer)));
  // The struct holds non-trivial members (chain_accum with std::vectors);
  // placement-new it so default constructors run, paired with an explicit
  // destructor call in destroy. iree_hal_command_buffer_initialize fills the
  // base resource header next, then the rest is set procedurally below.
  new (command_buffer) iree_hal_amdxdna_direct_command_buffer();
  iree_hal_command_buffer_initialize(
      device->device_allocator, mode, command_categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
      reinterpret_cast<uint8_t*>(command_buffer) + sizeof(*command_buffer),
      &iree_hal_amdxdna_direct_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device = device;
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
  // Run the destructor that pairs with the placement-new in create (releases
  // chain_accum's vector allocations + sub-command BOs).
  command_buffer->~iree_hal_amdxdna_direct_command_buffer();
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
    const iree_hal_buffer_barrier_t& barrier = buffer_barriers[i];
    const bool flush_host_to_device =
        iree_any_bit_set(barrier.source_scope,
                         IREE_HAL_ACCESS_SCOPE_HOST_WRITE |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE) &&
        iree_any_bit_set(barrier.target_scope,
                         IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
                             IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                             IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_READ);
    const bool invalidate_device_to_host =
        iree_any_bit_set(barrier.source_scope,
                         IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                             IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE) &&
        iree_any_bit_set(barrier.target_scope,
                         IREE_HAL_ACCESS_SCOPE_HOST_READ |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_READ);
    if (!flush_host_to_device && !invalidate_device_to_host) {
      continue;
    }
    if (IREE_UNLIKELY(!barrier.buffer_ref.buffer)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna direct command buffer cannot sync an indirect buffer "
          "barrier without a resolved buffer");
    }
    if (flush_host_to_device) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_buffer_flush_range(
          barrier.buffer_ref.buffer, barrier.buffer_ref.offset,
          barrier.buffer_ref.length));
    }
    if (invalidate_device_to_host) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_buffer_invalidate_range(
          barrier.buffer_ref.buffer, barrier.buffer_ref.offset,
          barrier.buffer_ref.length));
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

  const uint8_t* src =
      reinterpret_cast<const uint8_t*>(source_buffer) + source_offset;
  // No need to allocate scratch space (in an arena) as the memcpy
  // used below is expected to be synchronized.
  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(target_device_buffer,
                                             &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint8_t* dst =
      reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset;
  memcpy(dst, src, target_ref.length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              target_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
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
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(target_device_buffer,
                                             &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint8_t* dst =
      reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset;
  const iree_device_size_t length = target_ref.length;

  // Fast path for byte-pattern fills (most common case).
  if (pattern_length == 1) {
    memset(dst, *reinterpret_cast<const uint8_t*>(pattern), length);
  } else {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(pattern);
    for (iree_device_size_t i = 0; i < length; ++i) {
      dst[i] = p[i % pattern_length];
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              target_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
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
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(target_device_buffer,
                                             &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_hal_amdxdna_native_buffer_t* source_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(source_ref.buffer));
  void* source_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(source_device_buffer,
                                             &source_device_buffer_ptr));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;

  // Sync the host-mapped source range so the host memcpy reads device-written
  // data, then sync the target range back to device so a subsequent dispatch
  // sees the freshly copied bytes.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              source_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::device_to_host,
              target_ref.length, source_offset));
  memcpy(reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset,
         reinterpret_cast<uint8_t*>(source_device_buffer_ptr) + source_offset,
         target_ref.length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              target_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
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
namespace {
// TXN-interpreter selector: tells the firmware to interpret the instruction
// buffer as an XAie transaction (same value the default ERT_START_CU path
// passes as its opcode arg).
constexpr uint32_t kAie2ExecBufferKernelOpTxn = 3;

iree_status_t iree_hal_amdxdna_make_npu_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_cu_index_t cu_idx,
    const iree_hal_amdxdna_u32_list_t& txn,
    const iree_hal_amdxdna_u32_list_t& patches, const uint64_t* args,
    iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
    const iree_device_size_t* arg_offsets,
    const iree_device_size_t* arg_lengths, size_t arg_count,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    iree_hal_amdxdna_chain_cmd* out_cmd) {
  size_t bytes = txn.count * sizeof(uint32_t);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_alloc_buffer(
      command_buffer->device->native_device, bytes,
      iree_hal_amdxdna_native_buffer_type_t::instruction, &out_cmd->ctrl_code));
  void* mapped_ptr = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(
      out_cmd->ctrl_code.get(), &mapped_ptr));
  uint32_t* dst = static_cast<uint32_t*>(mapped_ptr);
  memcpy(dst, txn.data, bytes);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_patch_write32_constants(dst, txn.count, constants));
  if (!iree_hal_amdxdna_apply_patch_table(dst, txn.count, patches.data,
                                          patches.count, args, arg_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna cmd-chain: invalid host patch table for control code");
  }
  out_cmd->ctrl_words.assign(dst, dst + txn.count);
  if (arg_count) {
    out_cmd->binding_buffers.assign(arg_buffers, arg_buffers + arg_count);
    out_cmd->binding_device_addrs.resize(arg_count);
    for (size_t i = 0; i < arg_count; ++i) {
      out_cmd->binding_device_addrs[i] =
          iree_hal_amdxdna_native_buffer_device_address(arg_buffers[i]) +
          arg_offsets[i];
    }
    out_cmd->binding_offsets.assign(arg_offsets, arg_offsets + arg_count);
    out_cmd->binding_lengths.assign(arg_lengths, arg_lengths + arg_count);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
      out_cmd->ctrl_code.get(),
      iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  const iree_hal_amdxdna_native_command_opcode_t command_opcode =
      use_native_partial_elf
          ? iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf
          : iree_hal_amdxdna_native_command_opcode_t::start_npu;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      command_buffer->device->native_device, command_opcode,
      &out_cmd->command));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_set_cu_index(
      out_cmd->command.get(), cu_idx));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_add_control_buffer(
      out_cmd->command.get(), out_cmd->ctrl_code.get(), bytes));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_add_arg_32(
      out_cmd->command.get(), kAie2ExecBufferKernelOpTxn));
  const bool native_uses_dpu_regmap_args =
      !use_native_partial_elf &&
      command_buffer->device->native_caps.default_dispatch_opcode ==
          IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  if (command_opcode ==
      iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf) {
    if (IREE_UNLIKELY(arg_count &&
                      (!arg_buffers || !arg_offsets || !arg_lengths))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna PARTIAL_ELF cmd-chain child is missing BO "
          "bindings for its runtime args");
    }
    for (size_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
          out_cmd->command.get(), /*position=*/i + 1, arg_buffers[i],
          arg_offsets[i], arg_lengths[i]));
    }
  } else if (native_uses_dpu_regmap_args) {
    // Some native drivers expose DPU kernels through an xclbin XML register
    // map. In that path the runtime data VAs are regular ERT args.
    for (size_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_add_arg_64(
          out_cmd->command.get(), args[i]));
    }
  }
  out_cmd->built = true;
  return iree_ok_status();
}

template <typename T>
static bool iree_hal_amdxdna_span_equal(const std::vector<T>& lhs,
                                        const T* rhs, size_t rhs_count) {
  return lhs.size() == rhs_count &&
         (rhs_count == 0 || std::equal(lhs.begin(), lhs.end(), rhs));
}

static bool iree_hal_amdxdna_chain_cmd_matches_raw_descriptor(
    const iree_hal_amdxdna_chain_cmd& cmd,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patches,
    iree_hal_amdxdna_native_cu_index_t cu_idx,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    const uint64_t* args,
    iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
    const iree_device_size_t* arg_offsets,
    const iree_device_size_t* arg_lengths, size_t arg_count) {
  if (cmd.built || cmd.src_asm_inst != asm_inst ||
      cmd.src_patches != patches ||
      cmd.src_use_native_partial_elf != use_native_partial_elf ||
      cmd.src_cu_idx.index != cu_idx.index ||
      cmd.src_constants.size() != constants.data_length) {
    return false;
  }
  if (constants.data_length &&
      std::memcmp(cmd.src_constants.data(), constants.data,
                  constants.data_length) != 0) {
    return false;
  }
  return iree_hal_amdxdna_span_equal(cmd.binding_buffers, arg_buffers,
                                     arg_count) &&
         iree_hal_amdxdna_span_equal(cmd.binding_device_addrs, args,
                                     arg_count) &&
         iree_hal_amdxdna_span_equal(cmd.binding_offsets, arg_offsets,
                                     arg_count) &&
         iree_hal_amdxdna_span_equal(cmd.binding_lengths, arg_lengths,
                                     arg_count);
}

static iree_hal_amdxdna_chain_cmd
iree_hal_amdxdna_clone_unbuilt_chain_descriptor(
    const iree_hal_amdxdna_chain_cmd& src) {
  iree_hal_amdxdna_chain_cmd dst;
  dst.binding_buffers = src.binding_buffers;
  dst.binding_device_addrs = src.binding_device_addrs;
  dst.binding_offsets = src.binding_offsets;
  dst.binding_lengths = src.binding_lengths;
  dst.src_asm_inst = src.src_asm_inst;
  dst.src_patches = src.src_patches;
  dst.src_constants = src.src_constants;
  dst.src_cu_idx = src.src_cu_idx;
  dst.src_use_native_partial_elf = src.src_use_native_partial_elf;
  dst.repeat_count = 1;
  return dst;
}

static iree_status_t iree_hal_amdxdna_expand_repeated_chain_descriptors(
    iree_hal_amdxdna_chain_group& group) {
  const size_t logical_count =
      iree_hal_amdxdna_chain_group_logical_command_count(group);
  if (logical_count == group.cmds.size()) return iree_ok_status();
  std::vector<iree_hal_amdxdna_chain_cmd> expanded;
  expanded.reserve(logical_count);
  for (iree_hal_amdxdna_chain_cmd& cmd : group.cmds) {
    if (cmd.built && cmd.repeat_count > 1) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna cmd-chain cannot expand a repeated built descriptor");
    }
    const size_t repeat_count = std::max<size_t>(cmd.repeat_count, 1);
    cmd.repeat_count = 1;
    expanded.push_back(std::move(cmd));
    const iree_hal_amdxdna_chain_cmd& first = expanded.back();
    for (size_t i = 1; i < repeat_count; ++i) {
      expanded.push_back(
          iree_hal_amdxdna_clone_unbuilt_chain_descriptor(first));
    }
  }
  group.cmds = std::move(expanded);
  return iree_ok_status();
}
}  // namespace

// Accumulate one dispatch's reconfig+exec sub-commands into the command
// buffer's deferred-submit accumulator. Does NOT submit; flush_chains() chooses
// direct single-dispatch submit for one child and ERT_CMD_CHAIN for multi-child
// groups at end(). Dispatches that share a hw queue (e.g. all entry points of
// one control-packet executable, or separate executables resolved to the same
// shared context) accumulate into one group.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_context_ref_t* context_ref,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_cu_index_t cu_idx,
    iree_hal_amdxdna_kernel_params_t& kernel_params,
    iree_const_byte_span_t constants, bool use_native_partial_elf) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The chained path host-patches I/O addresses using the compiler-emitted
  // patch table (parallel to asm_inst_runlist). Require it: an executable
  // compiled before the patch table existed cannot use cmd-chain.
  if (kernel_params.patch_runlist_count !=
      kernel_params.asm_inst_runlist_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna cmd-chain requires a host patch table in the executable "
        "(have %zu patch lists for %zu control codes); recompile with a "
        "patch-table-aware compiler",
        kernel_params.patch_runlist_count,
        kernel_params.asm_inst_runlist_count);
  }

  // Binding device addresses (exec args). For control packets the reconfig arg
  // is the per-reconfiguration data buffer (built below).
  std::vector<uint64_t> binding_addrs(bindings.count);
  std::vector<iree_hal_amdxdna_native_buffer_t*> binding_buffers(
      bindings.count);
  std::vector<iree_device_size_t> binding_offsets(bindings.count);
  std::vector<iree_device_size_t> binding_lengths(bindings.count);
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_buffer_ensure_allocated(native_buffer));
    // Match the normal ERT_START_CU path: a binding may reference a subspan of
    // its allocated root BO, so host-patched DDR addresses must include both
    // offsets in addition to the BO base address.
    binding_buffers[j] = native_buffer;
    binding_offsets[j] =
        iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
        bindings.values[j].offset;
    binding_lengths[j] = bindings.values[j].length;
    binding_addrs[j] =
        iree_hal_amdxdna_native_buffer_device_address(native_buffer) +
        binding_offsets[j];
  }

  // Append to the current group, opening a new one when the native queue
  // changes (a chain runs on a single native context/queue).
  auto& groups = command_buffer->chain_accum.groups;
  if (groups.empty() || groups.back().queue != queue ||
      groups.back().native_partial_elf != use_native_partial_elf) {
    groups.emplace_back();
    groups.back().context =
        iree_hal_amdxdna_native_context_ref_retain(context_ref);
    groups.back().queue = queue;
    groups.back().native_partial_elf = use_native_partial_elf;
  }
  iree_hal_amdxdna_chain_group& group = groups.back();

  // Defer the per-child native build for the cacheable module-style chain path
  // (partial-ELF, no control-packet reconfiguration). Those children are
  // recorded as lightweight descriptors here and built lazily in flush only on
  // a cache miss, so a steady-state exact hit reuses the cached chain and skips
  // the build. Other paths (reconfiguration, non-partial-ELF) build eagerly.
  const bool defer_build =
      use_native_partial_elf && kernel_params.reconf_data_runlist_count == 0;

  // `run_idx` indexes both asm_inst_runlist and the parallel patch_runlist.
  auto emit = [&](size_t run_idx, const uint64_t* args,
                  iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
                  const iree_device_size_t* arg_offsets,
                  const iree_device_size_t* arg_lengths,
                  size_t arg_count) -> iree_status_t {
    if (defer_build && !group.cmds.empty() &&
        iree_hal_amdxdna_chain_cmd_matches_raw_descriptor(
            group.cmds.back(), &kernel_params.asm_inst_runlist[run_idx],
            &kernel_params.patch_runlist[run_idx], cu_idx, constants,
            use_native_partial_elf, args, arg_buffers, arg_offsets,
            arg_lengths, arg_count)) {
      if (IREE_UNLIKELY(group.cmds.back().repeat_count ==
                        std::numeric_limits<size_t>::max())) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "amdxdna cmd-chain repeat count overflow");
      }
      ++group.cmds.back().repeat_count;
      return iree_ok_status();
    }
    iree_hal_amdxdna_chain_cmd cmd;
    // Record the descriptor inputs (used for matching and lazy build).
    cmd.binding_buffers.assign(arg_buffers, arg_buffers + arg_count);
    cmd.binding_device_addrs.assign(args, args + arg_count);
    cmd.binding_offsets.assign(arg_offsets, arg_offsets + arg_count);
    cmd.binding_lengths.assign(arg_lengths, arg_lengths + arg_count);
    cmd.src_asm_inst = &kernel_params.asm_inst_runlist[run_idx];
    cmd.src_patches = &kernel_params.patch_runlist[run_idx];
    cmd.src_constants.assign(constants.data,
                             constants.data + constants.data_length);
    cmd.src_cu_idx = cu_idx;
    cmd.src_use_native_partial_elf = use_native_partial_elf;
    cmd.repeat_count = 1;
    if (!defer_build) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_make_npu_cmd(
          command_buffer, cu_idx, kernel_params.asm_inst_runlist[run_idx],
          kernel_params.patch_runlist[run_idx], args, arg_buffers, arg_offsets,
          arg_lengths, arg_count, constants, use_native_partial_elf, &cmd));
    }
    group.cmds.push_back(std::move(cmd));
    return iree_ok_status();
  };

  // Emit exactly one kernel slot for this HAL dispatch. Reconfiguration control
  // packets retain their repeat count for existing multi-PDI artifacts.
  size_t num_reconfigurations = kernel_params.reconf_data_runlist_count;
  if (num_reconfigurations == 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, emit(/*run_idx=*/0, binding_addrs.data(), binding_buffers.data(),
                 binding_offsets.data(), binding_lengths.data(),
                 bindings.count));
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Control-packet data buffer for this reconfiguration (reconfig arg[0]).
      iree_hal_amdxdna_u32_list_t& seq =
          kernel_params.reconf_data_runlist[i];
      size_t seq_bytes = seq.count * sizeof(uint32_t);
      iree_hal_amdxdna_native_buffer_ptr seq_buffer;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_amdxdna_native_device_alloc_buffer(
              command_buffer->device->native_device, seq_bytes,
              iree_hal_amdxdna_native_buffer_type_t::host_only, &seq_buffer));
      void* seq_buffer_ptr = nullptr;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_buffer_map(seq_buffer.get(),
                                                 &seq_buffer_ptr));
      memcpy(seq_buffer_ptr, seq.data, seq_bytes);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_buffer_sync_all(
                  seq_buffer.get(),
                  iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      group.reconf_buffers.push_back(std::move(seq_buffer));
      iree_hal_amdxdna_native_buffer_t* reconf_buffer =
          group.reconf_buffers.back().get();
      uint64_t reconf_arg =
          iree_hal_amdxdna_native_buffer_device_address(reconf_buffer);
      const iree_device_size_t reconf_offset = 0;
      const iree_device_size_t reconf_length =
          static_cast<iree_device_size_t>(seq_bytes);
      for (uint32_t r = 0; r < kernel_params.n_reconfigure_runs; r++) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, emit(/*run_idx=*/2 * i, &reconf_arg, &reconf_buffer,
                     &reconf_offset, &reconf_length, /*arg_count=*/1));
      }
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, emit(/*run_idx=*/2 * i + 1, binding_addrs.data(),
                   binding_buffers.data(), binding_offsets.data(),
                   binding_lengths.data(), bindings.count));
    }
  }

  // Track I/O bindings for residency + final device->host sync at flush.
  // Multiple HAL dispatches in the same command buffer often use the same
  // binding ranges; keep only exact ranges so a 240-dispatch chain does not
  // perform 240 duplicate host invalidations after the parent completes.
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    const iree_hal_buffer_ref_t binding_ref = bindings.values[j];
    const bool already_tracked =
        std::any_of(group.binding_refs.begin(), group.binding_refs.end(),
                    [&](const iree_hal_buffer_ref_t& existing) {
                      return existing.buffer == binding_ref.buffer &&
                             existing.offset == binding_ref.offset &&
                             existing.length == binding_ref.length;
                    });
    if (!already_tracked) group.binding_refs.push_back(binding_ref);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Build a native ERT_CMD_CHAIN for a contiguous span [begin, end) of a group's
// sub-commands, binding the group's referenced buffers for residency. The
// caller owns submission so native backends can submit all chunks before
// waiting when their DDI supports that shape.
static iree_status_t iree_hal_amdxdna_prepare_chain(
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_amdxdna_chain_group& group, size_t begin, size_t end,
    iree_hal_amdxdna_native_command_ptr* out_chain) {
  size_t n = end - begin;
  iree_hal_amdxdna_native_command_ptr chain;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      native_device, iree_hal_amdxdna_native_command_opcode_t::command_chain,
      &chain));
  std::vector<iree_hal_amdxdna_native_command_t*> commands;
  commands.reserve(n);
  for (size_t i = begin; i < end; ++i) {
    commands.push_back(group.cmds[i].command.get());
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_prepare_chain(
      chain.get(), commands.data(), commands.size()));

  const bool child_bo_table_parent_chain = group.native_partial_elf;
  if (child_bo_table_parent_chain) {
    *out_chain = std::move(chain);
    return iree_ok_status();
  }

  // Native address-list chain paths register every BO the firmware dereferences
  // (control code + control-packet data + I/O bindings) as arg BOs on the
  // submitted chain so the driver keeps them resident; the sub-command slots
  // reference them only by address. Module-style partial-ELF chains
  // intentionally skip this superset above: the parent binds child exec BOs
  // only, while each child command BO carries its own BO table.
  const size_t arg_bo_ceiling =
      iree_hal_amdxdna_native_command_arg_binding_capacity();
  size_t arg_total =
      n + group.reconf_buffers.size() + group.binding_refs.size();
  if (arg_total > arg_bo_ceiling) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna cmd-chain: %zu arg BOs exceeds native ceiling %zu (chunk "
        "groups or reduce binding count)",
        arg_total, arg_bo_ceiling);
  }
  size_t arg_pos = 0;
  for (size_t i = begin; i < end; i++) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
        chain.get(), arg_pos++, group.cmds[i].ctrl_code.get(), 0,
        iree_hal_amdxdna_native_buffer_size(group.cmds[i].ctrl_code.get())));
  }
  for (iree_hal_amdxdna_native_buffer_ptr& seq_buffer : group.reconf_buffers) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
        chain.get(), arg_pos++, seq_buffer.get(), 0,
        iree_hal_amdxdna_native_buffer_size(seq_buffer.get())));
  }
  for (const iree_hal_buffer_ref_t& binding_ref : group.binding_refs) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(binding_ref.buffer));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
        chain.get(), arg_pos++, native_buffer, 0,
        iree_hal_amdxdna_native_buffer_size(native_buffer)));
  }

  *out_chain = std::move(chain);
  return iree_ok_status();
}

static bool iree_hal_amdxdna_chain_group_requires_parent_chain(
    const iree_hal_amdxdna_chain_group& group) {
  return iree_hal_amdxdna_chain_group_logical_command_count(group) > 1 ||
         !group.reconf_buffers.empty();
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_submit_accumulated_single(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group& group) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (group.cmds.size() != 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "single-dispatch flush expected one command, got "
                            "%zu",
                            group.cmds.size());
  }
  iree_hal_amdxdna_chain_cmd& cmd = group.cmds[0];
  if (!cmd.src_asm_inst || !cmd.src_patches) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "single-dispatch flush is missing recorded control-code descriptors");
  }

  if (!cmd.src_use_native_partial_elf) {
    if (!cmd.built) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_make_npu_cmd(
                  command_buffer, cmd.src_cu_idx, *cmd.src_asm_inst,
                  *cmd.src_patches, cmd.binding_device_addrs.data(),
                  cmd.binding_buffers.data(), cmd.binding_offsets.data(),
                  cmd.binding_lengths.data(), cmd.binding_device_addrs.size(),
                  iree_make_const_byte_span(cmd.src_constants.data(),
                                            cmd.src_constants.size()),
                  cmd.src_use_native_partial_elf, &cmd));
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_queue_submit_and_wait(
                group.queue, cmd.command.get(), IREE_SV("dispatch")));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  std::vector<uint32_t> prepared_ctrl_words(
      cmd.src_asm_inst->data, cmd.src_asm_inst->data + cmd.src_asm_inst->count);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_patch_write32_constants(
              prepared_ctrl_words.data(), prepared_ctrl_words.size(),
              iree_make_const_byte_span(cmd.src_constants.data(),
                                        cmd.src_constants.size())));
  if (!iree_hal_amdxdna_apply_patch_table(
          prepared_ctrl_words.data(), prepared_ctrl_words.size(),
          cmd.src_patches->data, cmd.src_patches->count,
          cmd.binding_device_addrs.data(),
          cmd.binding_device_addrs.size())) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna PARTIAL_ELF single dispatch has an invalid host patch table");
  }

  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache =
      iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
  if (!single_command_cache) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to allocate amdxdna single command cache");
  }
  AmdxdnaSlimMutexUniqueLock single_cache_lock(&single_command_cache->mutex);
  iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_find_single_command_cache_entry(
              single_command_cache, group.queue, cmd.src_cu_idx.index,
              prepared_ctrl_words.data(), prepared_ctrl_words.size(),
              cmd.binding_buffers.data(), cmd.binding_device_addrs.data(),
              cmd.binding_offsets.data(), cmd.binding_lengths.data(),
              cmd.binding_device_addrs.size(), &single_cache_entry));

  iree_hal_amdxdna_native_command_t* submit_command = nullptr;
  if (single_cache_entry) {
    submit_command = single_cache_entry->command;
  } else {
    const size_t ctrl_code_size =
        prepared_ctrl_words.size() * sizeof(uint32_t);
    iree_hal_amdxdna_native_buffer_ptr ctrl_code_buffer;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_device_alloc_buffer(
                command_buffer->device->native_device, ctrl_code_size,
                iree_hal_amdxdna_native_buffer_type_t::instruction,
                &ctrl_code_buffer));
    void* instr_buffer_ptr = nullptr;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_buffer_map(ctrl_code_buffer.get(),
                                               &instr_buffer_ptr));
    memcpy(instr_buffer_ptr, prepared_ctrl_words.data(), ctrl_code_size);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_buffer_sync_all(
                ctrl_code_buffer.get(),
                iree_hal_amdxdna_native_sync_direction_t::host_to_device));

    iree_hal_amdxdna_native_command_ptr command;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_create(
                command_buffer->device->native_device,
                iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf,
                &command));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_set_cu_index(command.get(),
                                                         cmd.src_cu_idx));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_control_buffer(
                command.get(), ctrl_code_buffer.get(), ctrl_code_size));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_arg_32(
                command.get(), kAie2ExecBufferKernelOpTxn));
    for (size_t i = 0; i < cmd.binding_buffers.size(); ++i) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_bind_buffer(
                  command.get(), /*position=*/i + 1, cmd.binding_buffers[i],
                  cmd.binding_offsets[i], cmd.binding_lengths[i]));
    }
    std::vector<iree_hal_amdxdna_native_buffer_t*> binding_buffers =
        cmd.binding_buffers;
    std::vector<uint64_t> binding_device_addrs = cmd.binding_device_addrs;
    std::vector<iree_device_size_t> binding_offsets = cmd.binding_offsets;
    std::vector<iree_device_size_t> binding_lengths = cmd.binding_lengths;
    single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
        single_command_cache, group.queue, cmd.src_cu_idx.index,
        prepared_ctrl_words.data(), prepared_ctrl_words.size(),
        binding_buffers.data(), binding_device_addrs.data(),
        binding_offsets.data(), binding_lengths.data(),
        binding_device_addrs.size(), ctrl_code_buffer.get(), command.get());
    if (!single_cache_entry) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to store amdxdna single command cache "
                              "entry");
    }
    (void)ctrl_code_buffer.release();
    (void)command.release();
    submit_command = single_cache_entry->command;
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_queue_submit_and_wait(group.queue,
                                                        submit_command,
                                                        IREE_SV("dispatch")));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Flush all accumulated groups. Single-child groups submit directly; groups
// with multiple children become native ERT chains chunked to the backend slot
// limit. Groups are submitted in recorded order so producer/consumer
// dependencies across groups are honored by the device's in-order completion.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_flush_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer) {
  auto& groups = command_buffer->chain_accum.groups;
  if (groups.empty()) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  const bool has_parent_chain_group = std::any_of(
      groups.begin(), groups.end(),
      [](const iree_hal_amdxdna_chain_group& group) {
        return iree_hal_amdxdna_chain_group_requires_parent_chain(group);
      });

  // Max slots per chain that fit the fixed-size exec buffer (constant per
  // device; computed once and cached). Atomic load with relaxed ordering: a
  // racing first-time probe is idempotent (same value), and the slot count is
  // independent data so we don't need ordering against any other state. Use
  // acquire on the success path / release on the store so a thread observing
  // the cached value also observes the probe's published writes.
  uint32_t max_slots = std::numeric_limits<uint32_t>::max();
  if (has_parent_chain_group) {
    max_slots = iree_atomic_load(&command_buffer->device->chain_max_slots,
                                 iree_memory_order_acquire);
    if (max_slots == 0) {
      max_slots = command_buffer->device->native_caps.max_command_chain_slots;
      if (max_slots == 0) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_native_device_query_chain_max_slots(
                    command_buffer->device->native_device, &max_slots));
      }
      iree_atomic_store(&command_buffer->device->chain_max_slots, max_slots,
                        iree_memory_order_release);
    }
  }

  // Submit each accumulated group either as one direct native command or as
  // native ERT chains chunked into max_slots-sized pieces.
  iree_status_t status = iree_ok_status();
  for (iree_hal_amdxdna_chain_group& group : groups) {
    const bool submit_as_chain =
        iree_hal_amdxdna_chain_group_requires_parent_chain(group);
    if (!submit_as_chain) {
      status =
          iree_hal_amdxdna_direct_command_buffer_submit_accumulated_single(
              command_buffer, group);
    } else {
      std::unique_lock<std::mutex> chain_cache_lock;
      iree_hal_amdxdna_chain_command_cache_entry* chain_cache = nullptr;
      if (group.native_partial_elf && group.reconf_buffers.empty()) {
        iree_hal_amdxdna_device_chain_command_cache_t* device_chain_cache =
            iree_hal_amdxdna_get_chain_command_cache(command_buffer->device);
        chain_cache_lock =
            std::unique_lock<std::mutex>(device_chain_cache->mutex);
        auto rebuild_cached_parent_chains = [&]() -> iree_status_t {
          chain_cache->chains.clear();
          for (size_t begin = 0; begin < chain_cache->group.cmds.size();
               begin += max_slots) {
            size_t end = std::min<size_t>(begin + max_slots,
                                          chain_cache->group.cmds.size());
            iree_hal_amdxdna_native_command_ptr chain;
            IREE_RETURN_IF_ERROR(iree_hal_amdxdna_prepare_chain(
                command_buffer->device->native_device, chain_cache->group,
                begin, end, &chain));
            chain_cache->chains.push_back(std::move(chain));
          }
          return iree_ok_status();
        };
        auto touch_chain_cache_entry = [&]() {
          chain_cache->last_use = ++device_chain_cache->use_clock;
        };
        bool exact_cache_hit = false;
        bool device_cache_hit = false;
        bool group_expanded = false;
        auto expand_group_once = [&]() -> iree_status_t {
          if (group_expanded) return iree_ok_status();
          IREE_RETURN_IF_ERROR(
              iree_hal_amdxdna_expand_repeated_chain_descriptors(group));
          group_expanded = true;
          return iree_ok_status();
        };
        // Deferred-build fast path: reuse an already-built cached chain when
        // the descriptor inputs (control-code template + constants + bindings)
        // match exactly, without building this group's children at all.
        for (iree_hal_amdxdna_chain_command_cache_entry& entry :
             device_chain_cache->entries) {
          if (iree_hal_amdxdna_chain_command_cache_descriptor_matches(
                  entry, group, max_slots)) {
            chain_cache = &entry;
            exact_cache_hit = true;
            break;
          }
        }
        // Not an exact hit: realize any deferred children now so the
        // ctrl_words-based device/shape/miss logic below can match, update, or
        // cache them.
        if (!chain_cache) {
          status = expand_group_once();
        }
        if (!chain_cache && iree_status_is_ok(status)) {
          for (iree_hal_amdxdna_chain_cmd& cmd : group.cmds) {
            if (cmd.built) continue;
            status = iree_hal_amdxdna_make_npu_cmd(
                command_buffer, cmd.src_cu_idx, *cmd.src_asm_inst,
                *cmd.src_patches, cmd.binding_device_addrs.data(),
                cmd.binding_buffers.data(), cmd.binding_offsets.data(),
                cmd.binding_lengths.data(), cmd.binding_device_addrs.size(),
                iree_make_const_byte_span(cmd.src_constants.data(),
                                          cmd.src_constants.size()),
                cmd.src_use_native_partial_elf, &cmd);
            if (!iree_status_is_ok(status)) break;
          }
        }
        if (chain_cache) {
          touch_chain_cache_entry();
        } else if (iree_status_is_ok(status)) {
          for (iree_hal_amdxdna_chain_command_cache_entry& entry :
               device_chain_cache->entries) {
            if (iree_hal_amdxdna_chain_command_cache_device_matches(
                    entry, group, max_slots)) {
              chain_cache = &entry;
              break;
            }
          }
          if (chain_cache) {
            device_cache_hit = true;
            touch_chain_cache_entry();
          }
        }
        if (iree_status_is_ok(status) && !chain_cache &&
            device_chain_cache->entries.size() >=
                kAmdxdnaChainCommandCacheCapacity) {
          for (iree_hal_amdxdna_chain_command_cache_entry& entry :
               device_chain_cache->entries) {
            if (iree_hal_amdxdna_chain_command_cache_shape_matches(entry, group,
                                                                   max_slots)) {
              chain_cache = &entry;
              break;
            }
          }
        }
        if (chain_cache && !exact_cache_hit && !device_cache_hit) {
          touch_chain_cache_entry();
          bool packet_changed = false;
          for (size_t i = 0; i < group.cmds.size() && iree_status_is_ok(status);
               ++i) {
            bool cmd_packet_changed = false;
            status = iree_hal_amdxdna_update_cached_chain_cmd(
                chain_cache->group.cmds[i], group.cmds[i], &cmd_packet_changed,
                nullptr, nullptr, nullptr);
            if (cmd_packet_changed) {
              packet_changed = true;
            }
          }
          if (packet_changed) {
            for (iree_hal_amdxdna_native_command_ptr& chain :
                 chain_cache->chains) {
              if (!iree_status_is_ok(status)) break;
              status = iree_hal_amdxdna_native_command_mark_chain_code_dirty(
                  chain.get());
            }
          }
        } else if (iree_status_is_ok(status) && !chain_cache) {
          const size_t cache_capacity = kAmdxdnaChainCommandCacheCapacity;
          if (device_chain_cache->entries.size() < cache_capacity) {
            device_chain_cache->entries.emplace_back();
            chain_cache = &device_chain_cache->entries.back();
          } else {
            auto victim_it = std::min_element(
                device_chain_cache->entries.begin(),
                device_chain_cache->entries.end(),
                [](const iree_hal_amdxdna_chain_command_cache_entry& lhs,
                   const iree_hal_amdxdna_chain_command_cache_entry& rhs) {
                  return lhs.last_use < rhs.last_use;
                });
            chain_cache = &*victim_it;
          }
          chain_cache->chains.clear();
          iree_hal_amdxdna_native_context_ref_release(
              chain_cache->group.context);
          chain_cache->group.context =
              iree_hal_amdxdna_native_context_ref_retain(group.context);
          chain_cache->group.queue = group.queue;
          chain_cache->group.native_partial_elf = group.native_partial_elf;
          chain_cache->group.cmds = std::move(group.cmds);
          chain_cache->group.reconf_buffers.clear();
          chain_cache->group.binding_refs.clear();
          chain_cache->max_slots = max_slots;
          touch_chain_cache_entry();
          if (iree_status_is_ok(status)) {
            status = rebuild_cached_parent_chains();
          }
        }
        if (iree_status_is_ok(status) && !chain_cache->chains.empty()) {
          std::vector<iree_hal_amdxdna_native_command_t*> chain_ptrs;
          chain_ptrs.reserve(chain_cache->chains.size());
          for (iree_hal_amdxdna_native_command_ptr& chain :
               chain_cache->chains) {
            chain_ptrs.push_back(chain.get());
          }
          status = iree_hal_amdxdna_native_queue_submit_all_and_wait(
              group.queue, chain_ptrs.data(), chain_ptrs.size(),
              IREE_SV("ERT_CMD_CHAIN"));
        }
      } else {
        std::vector<iree_hal_amdxdna_native_command_ptr> chains;
        for (size_t begin = 0;
             begin < group.cmds.size() && iree_status_is_ok(status);
             begin += max_slots) {
          size_t end = std::min(begin + max_slots, group.cmds.size());
          iree_hal_amdxdna_native_command_ptr chain;
          status = iree_hal_amdxdna_prepare_chain(
              command_buffer->device->native_device, group, begin, end, &chain);
          if (iree_status_is_ok(status)) chains.push_back(std::move(chain));
        }
        if (iree_status_is_ok(status) && !chains.empty()) {
          std::vector<iree_hal_amdxdna_native_command_t*> chain_ptrs;
          chain_ptrs.reserve(chains.size());
          for (iree_hal_amdxdna_native_command_ptr& chain : chains) {
            chain_ptrs.push_back(chain.get());
          }
          status = iree_hal_amdxdna_native_queue_submit_all_and_wait(
              group.queue, chain_ptrs.data(), chain_ptrs.size(),
              IREE_SV("ERT_CMD_CHAIN"));
        }
      }
    }
    if (!iree_status_is_ok(status)) break;
    // Parent chains do not go through the native submit path's normal binding
    // sync model, so sync this group's I/O bindings back to host once the
    // chains complete. The direct single path mirrors normal_run() and relies
    // on the native submit's binding sync behavior.
    if (submit_as_chain) {
      for (const iree_hal_buffer_ref_t& binding_ref : group.binding_refs) {
        status = iree_hal_amdxdna_buffer_invalidate_range(
            binding_ref.buffer, binding_ref.offset, binding_ref.length);
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
  groups.clear();

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_normal_run(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_cu_index_t cu_idx,
    const iree_hal_amdxdna_u32_list_t& asm_inst,
    const iree_hal_amdxdna_u32_list_t* patch_table,
    iree_const_byte_span_t constants,
    bool use_single_partial_elf) {
  IREE_TRACE_ZONE_BEGIN(z0);

  std::vector<uint64_t> binding_addrs;
  std::vector<iree_hal_amdxdna_native_buffer_t*> binding_buffers;
  std::vector<iree_device_size_t> binding_offsets;
  std::vector<iree_device_size_t> binding_lengths;
  if (use_single_partial_elf) {
    binding_addrs.resize(bindings.count);
    binding_buffers.reserve(bindings.count);
    binding_offsets.reserve(bindings.count);
    binding_lengths.reserve(bindings.count);
    for (iree_host_size_t j = 0; j < bindings.count; ++j) {
      iree_hal_amdxdna_native_buffer_t* native_buffer =
          iree_hal_amdxdna_buffer_handle(
              iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_buffer_ensure_allocated(native_buffer));
      const iree_device_size_t native_offset =
          iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
          bindings.values[j].offset;
      binding_addrs[j] =
          iree_hal_amdxdna_native_buffer_device_address(native_buffer) +
          native_offset;
      binding_buffers.push_back(native_buffer);
      binding_offsets.push_back(native_offset);
      binding_lengths.push_back(bindings.values[j].length);
    }
  }

  std::vector<uint32_t> prepared_ctrl_words;
  iree_hal_amdxdna_native_command_t* submit_command = nullptr;
  // Production policy: reuse a prepared single-dispatch native command across
  // queue_execute calls (keyed by the dispatch signature in the device single-
  // command cache) instead of rebuilding it each time. Always on; kept as a
  // named flag until the prepared-command model in the native-DDI follow-ups
  // replaces the device-global caches.
  const bool use_single_command_cache = true;
  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache =
      nullptr;
  iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = nullptr;
  AmdxdnaSlimMutexUniqueLock single_cache_lock;
  if (use_single_partial_elf) {
    prepared_ctrl_words.assign(asm_inst.data, asm_inst.data + asm_inst.count);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdxdna_patch_write32_constants(
            prepared_ctrl_words.data(), prepared_ctrl_words.size(), constants));
    if (!iree_hal_amdxdna_apply_patch_table(
            prepared_ctrl_words.data(), prepared_ctrl_words.size(),
            patch_table->data, patch_table->count, binding_addrs.data(),
            binding_addrs.size())) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna PARTIAL_ELF single dispatch has an invalid host "
          "patch table");
    }
    if (use_single_command_cache) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "failed to allocate amdxdna single command cache");
      }
      single_cache_lock.lock(&single_command_cache->mutex);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_find_single_command_cache_entry(
                  single_command_cache, queue, cu_idx.index,
                  prepared_ctrl_words.data(), prepared_ctrl_words.size(),
                  binding_buffers.data(), binding_addrs.data(),
                  binding_offsets.data(), binding_lengths.data(),
                  binding_addrs.size(), &single_cache_entry));
      if (single_cache_entry) {
        submit_command = single_cache_entry->command;
      } else {
        single_cache_lock.unlock();
      }
    }
  }

  // Allocate a buffer object to hold the control code (`asm_inst`).
  size_t ctrl_code_size =
      (use_single_partial_elf ? prepared_ctrl_words.size() : asm_inst.count) *
      sizeof(uint32_t);
  iree_hal_amdxdna_native_buffer_ptr ctrl_code_buffer;
  if (!submit_command) {
    const bool uses_native_instruction_buffer =
        use_single_partial_elf ||
        (command_buffer->device->native_caps.dispatch_models &
         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU) != 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_device_alloc_buffer(
                command_buffer->device->native_device, ctrl_code_size,
                uses_native_instruction_buffer
                    ? iree_hal_amdxdna_native_buffer_type_t::instruction
                    : iree_hal_amdxdna_native_buffer_type_t::cacheable,
                &ctrl_code_buffer));
  }
  void* instr_buffer_ptr = nullptr;
  uint32_t* instr_buffer = nullptr;
  if (!submit_command) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_buffer_map(ctrl_code_buffer.get(),
                                               &instr_buffer_ptr));
    instr_buffer = static_cast<uint32_t*>(instr_buffer_ptr);
    if (use_single_partial_elf) {
      memcpy(instr_buffer, prepared_ctrl_words.data(), ctrl_code_size);
    } else {
      memcpy(instr_buffer, asm_inst.data, ctrl_code_size);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_patch_write32_constants(
                  instr_buffer, asm_inst.count, constants));
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_buffer_sync_all(
                ctrl_code_buffer.get(),
                iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  }

  iree_hal_amdxdna_native_command_ptr command;
  const iree_hal_amdxdna_native_command_opcode_t command_opcode =
      use_single_partial_elf
          ? iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf
          : iree_hal_amdxdna_native_command_opcode_from_c(
                command_buffer->device->native_caps.default_dispatch_opcode);
  if (!submit_command) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdxdna_native_command_create(
            command_buffer->device->native_device, command_opcode, &command));
    // Add the kernel arguments.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdxdna_native_command_set_cu_index(command.get(), cu_idx));
    if (use_single_partial_elf) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_control_buffer(
                  command.get(), ctrl_code_buffer.get(), ctrl_code_size));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_arg_32(
                  command.get(), kAie2ExecBufferKernelOpTxn));
    } else if ((command_buffer->device->native_caps.dispatch_models &
                IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU) != 0) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_control_buffer(
                  command.get(), ctrl_code_buffer.get(), ctrl_code_size));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_arg_32(
                  command.get(), kAie2ExecBufferKernelOpTxn));
    } else {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_arg_64(
                  command.get(), kAmdxdnaControlCodeOpcode));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_buffer_arg(
                  command.get(), ctrl_code_buffer.get()));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_command_add_arg_32(
                  command.get(), (uint32_t)asm_inst.count));
    }
  }

  if (!submit_command) {
    if (use_single_partial_elf) {
      for (iree_host_size_t j = 0; j < bindings.count; ++j) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_native_command_bind_buffer(
                    command.get(), /*position=*/j + 1, binding_buffers[j],
                    binding_offsets[j], binding_lengths[j]));
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
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
                    command.get(), native_buffer, native_offset));
      }
    }
  }

  if (use_single_partial_elf && use_single_command_cache && !submit_command) {
    if (!single_command_cache) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "failed to allocate amdxdna single command cache");
      }
    }
    if (!single_cache_lock.owns_lock()) {
      single_cache_lock.lock(&single_command_cache->mutex);
    }
    // Another queue worker may have populated the entry while this thread was
    // building the native command. Recheck under the cache lock before storing.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_find_single_command_cache_entry(
                single_command_cache, queue, cu_idx.index,
                prepared_ctrl_words.data(), prepared_ctrl_words.size(),
                binding_buffers.data(), binding_addrs.data(),
                binding_offsets.data(), binding_lengths.data(),
                binding_addrs.size(), &single_cache_entry));
    if (!single_cache_entry) {
      single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
          single_command_cache, queue, cu_idx.index,
          prepared_ctrl_words.data(), prepared_ctrl_words.size(),
          binding_buffers.data(), binding_addrs.data(), binding_offsets.data(),
          binding_lengths.data(), binding_addrs.size(), ctrl_code_buffer.get(),
          command.get());
      if (!single_cache_entry) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to store amdxdna single command cache "
                                "entry");
      }
      (void)ctrl_code_buffer.release();
      (void)command.release();
    }
    submit_command = single_cache_entry->command;
  }

  if (!submit_command) {
    submit_command = command.get();
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_queue_submit_and_wait(queue, submit_command,
                                                        IREE_SV("dispatch")));
  // Sync the bindings back to the host.
  if (!use_single_partial_elf &&
      command_buffer->device->native_caps.buffer_sync_model !=
          IREE_HAL_AMDXDNA_NATIVE_C_BUFFER_SYNC_MODEL_SUBMIT_SYNCS_BINDINGS) {
    for (iree_host_size_t j = 0; j < bindings.count; ++j) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_buffer_invalidate_range(
                  bindings.values[j].buffer, bindings.values[j].offset,
                  bindings.values[j].length));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reconfigure(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_cu_index_t cu_idx, uint32_t n_reconfigure_runs,
    const iree_hal_amdxdna_u32_list_t& ctrlpkt_inst,
    const iree_hal_amdxdna_u32_list_t& ctrlpkt_seq,
    iree_const_byte_span_t constants) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Allocate a buffer object to hold the control packet instructions.
  size_t ctrlpkt_inst_size = ctrlpkt_inst.count * sizeof(uint32_t);
  iree_hal_amdxdna_native_buffer_ptr ctrlpkt_inst_buffer;
  const bool uses_native_instruction_buffer =
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU) != 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_device_alloc_buffer(
              command_buffer->device->native_device, ctrlpkt_inst_size,
              uses_native_instruction_buffer
                  ? iree_hal_amdxdna_native_buffer_type_t::instruction
                  : iree_hal_amdxdna_native_buffer_type_t::cacheable,
              &ctrlpkt_inst_buffer));
  void* ctrlpkt_inst_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(ctrlpkt_inst_buffer.get(),
                                             &ctrlpkt_inst_ptr));
  auto* ctrlpkt_inst_words = static_cast<uint32_t*>(ctrlpkt_inst_ptr);
  memcpy(ctrlpkt_inst_words, ctrlpkt_inst.data, ctrlpkt_inst_size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_patch_write32_constants(
              ctrlpkt_inst_words, ctrlpkt_inst.count, constants));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync_all(
              ctrlpkt_inst_buffer.get(),
              iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  // Allocate a buffer object to hold the control packet sequence (content).
  size_t ctrlpkt_seq_size = ctrlpkt_seq.count * sizeof(uint32_t);
  iree_hal_amdxdna_native_buffer_ptr ctrlpkt_seq_buffer;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_device_alloc_buffer(
              command_buffer->device->native_device, ctrlpkt_seq_size,
              iree_hal_amdxdna_native_buffer_type_t::host_only,
              &ctrlpkt_seq_buffer));
  void* ctrlpkt_seq_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(ctrlpkt_seq_buffer.get(),
                                             &ctrlpkt_seq_ptr));
  memcpy(ctrlpkt_seq_ptr, ctrlpkt_seq.data, ctrlpkt_seq_size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync_all(
              ctrlpkt_seq_buffer.get(),
              iree_hal_amdxdna_native_sync_direction_t::host_to_device));

  iree_hal_amdxdna_native_command_ptr command;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_create(
              command_buffer->device->native_device,
              iree_hal_amdxdna_native_command_opcode_from_c(
                  command_buffer->device->native_caps.default_dispatch_opcode),
              &command));
  // Add the kernel arguments.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_set_cu_index(command.get(), cu_idx));
  if ((command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU) != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_control_buffer(
                command.get(), ctrlpkt_inst_buffer.get(), ctrlpkt_inst_size));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_arg_32(
                command.get(), kAie2ExecBufferKernelOpTxn));
  } else {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_arg_64(
                command.get(), kAmdxdnaControlCodeOpcode));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_buffer_arg(
                command.get(), ctrlpkt_inst_buffer.get()));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_arg_32(
                command.get(), (uint32_t)ctrlpkt_inst.count));
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_buffer_arg(
              command.get(), ctrlpkt_seq_buffer.get()));
  // Execute the reconfiguration for `n_reconfigure_runs` times.
  for (int i = 0; i < n_reconfigure_runs; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdxdna_native_queue_submit_and_wait(
            queue, command.get(), IREE_SV("control-packet reconfiguration")));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

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
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point function %" PRIu64
                            " out of range; executable only contains %" PRIhsz
                            " entry points",
                            function.value, executable->entry_point_count);
  }
  const uint32_t entry_point = iree_hal_executable_function_index(function);
  if (entry_point >= executable->entry_point_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %u out of range; executable "
                            "only contains %" PRIhsz " entry points",
                            entry_point, executable->entry_point_count);
  }
  iree_hal_amdxdna_kernel_params_t& kernel_params =
      executable->entry_points[entry_point];
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_validate_live_dispatch_bindings(bindings));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  size_t num_reconfigurations = kernel_params.reconf_data_runlist_count;
  iree_const_byte_span_t pdi_span = iree_make_const_byte_span(
      kernel_params.pdi.data, kernel_params.pdi.count);
  iree_const_byte_span_t xclbin_span = iree_make_const_byte_span(
      kernel_params.xclbin.data, kernel_params.xclbin.count);
  iree_string_view_t kernel_name = kernel_params.kernel_name;
  const iree_hal_amdxdna_u32_list_t* single_patch_table =
      kernel_params.patch_runlist_count == 0 ? nullptr
                                             : &kernel_params.patch_runlist[0];
  const bool use_native_partial_elf_context =
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF) != 0 &&
      num_reconfigurations == 0 && xclbin_span.data_length > 0 &&
      iree_hal_amdxdna_patch_table_is_valid(single_patch_table);
  std::unique_ptr<iree_hal_amdxdna_native_context_ref_t,
                  decltype(&iree_hal_amdxdna_native_context_ref_release)>
      context_ref(nullptr, iree_hal_amdxdna_native_context_ref_release);
  iree_hal_amdxdna_native_cu_index_t cu_idx;
  if (num_reconfigurations == 0) {
    if (use_native_partial_elf_context) {
      // Memoize the resolved context + CU on the entry point. Otherwise every
      // dispatch re-enters get_or_create_context, which copies and FNV-hashes
      // the whole PDI under a lock; that dominates a 240-dispatch chain. Repeat
      // dispatches reuse the cached context and skip the CU open entirely.
      AmdxdnaSlimMutexLock lock(&executable->context_mutex);
      if (kernel_params.cached_context_valid) {
        context_ref.reset(iree_hal_amdxdna_native_context_ref_retain(
            kernel_params.cached_context));
        cu_idx = iree_hal_amdxdna_native_cu_index_from_c(
            kernel_params.cached_cu_index);
      } else {
        iree_hal_amdxdna_native_context_ref_t* raw_context_ref = nullptr;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_device_get_or_create_context(
                    command_buffer->device, pdi_span, xclbin_span, kernel_name,
                    &raw_context_ref));
        context_ref.reset(raw_context_ref);
        iree_hal_amdxdna_native_c_cu_index_t c_cu_idx;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_native_context_ref_open_cu(
                    context_ref.get(), kernel_name, &c_cu_idx));
        cu_idx = iree_hal_amdxdna_native_cu_index_from_c(c_cu_idx);
        kernel_params.cached_context =
            iree_hal_amdxdna_native_context_ref_retain(context_ref.get());
        kernel_params.cached_cu_index = c_cu_idx;
        kernel_params.cached_context_valid = true;
      }
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
      iree_hal_amdxdna_native_context_ref_t* raw_context_ref = nullptr;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_device_c_create_context_ref(
                  command_buffer->device->native_device, &context_image,
                  &raw_context_ref));
      context_ref.reset(raw_context_ref);
      iree_hal_amdxdna_native_c_cu_index_t c_cu_idx;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_context_ref_open_cu(
                  context_ref.get(), kernel_name, &c_cu_idx));
      cu_idx = iree_hal_amdxdna_native_cu_index_from_c(c_cu_idx);
    }
  } else if (kernel_params.pdi.count != 0 || kernel_params.xclbin.count != 0) {
    iree_hal_amdxdna_native_context_ref_t* raw_context_ref = nullptr;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_device_get_or_create_context(
                command_buffer->device, pdi_span, xclbin_span, kernel_name,
                &raw_context_ref));
    context_ref.reset(raw_context_ref);
    iree_hal_amdxdna_native_c_cu_index_t c_cu_idx;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_context_ref_open_cu(
                context_ref.get(), kernel_name, &c_cu_idx));
    cu_idx = iree_hal_amdxdna_native_cu_index_from_c(c_cu_idx);
    {
      AmdxdnaSlimMutexLock lock(&executable->context_mutex);
      iree_hal_amdxdna_native_context_ref_release(executable->context);
      executable->context =
          iree_hal_amdxdna_native_context_ref_retain(context_ref.get());
      executable->context_cu_index = c_cu_idx;
      executable->context_cu_index_valid = true;
    }
  } else {
    AmdxdnaSlimMutexLock lock(&executable->context_mutex);
    if (executable->context_cu_index_valid) {
      context_ref.reset(
          iree_hal_amdxdna_native_context_ref_retain(executable->context));
      cu_idx = iree_hal_amdxdna_native_cu_index_from_c(
          executable->context_cu_index);
    }
  }
  if (!context_ref) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna: control-packet dispatch with no context image ran before "
        "its PDI/xclbin-carrying entry point loaded the array");
  }
  iree_hal_amdxdna_native_queue_t* queue =
      iree_hal_amdxdna_native_context_ref_queue(context_ref.get());

  const bool has_host_patch_table =
      kernel_params.patch_runlist_count ==
      kernel_params.asm_inst_runlist_count;
  const bool multi_control_code_or_pdi =
      kernel_params.n_pdi_loads > 1 ||
      kernel_params.asm_inst_runlist_count > 1 || num_reconfigurations != 0;
  const bool use_deferred_submit_policy =
      has_host_patch_table &&
      (use_native_partial_elf_context || multi_control_code_or_pdi);
  if (use_deferred_submit_policy) {
    // Accumulate this dispatch's commands; end() chooses the native shape from
    // the recorded stream: one child submits as a direct single dispatch, while
    // multiple children or multi-control-code artifacts become ERT_CMD_CHAINs.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
                bindings, command_buffer, context_ref.get(), queue, cu_idx,
                kernel_params, constants, use_native_partial_elf_context));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (num_reconfigurations == 0) {
    // Normal kernel dispatch.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_direct_command_buffer_normal_run(
                bindings, command_buffer, queue, cu_idx,
                kernel_params.asm_inst_runlist[0],
                single_patch_table, constants, use_native_partial_elf_context));
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Reconfigure the device.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_amdxdna_direct_command_buffer_reconfigure(
              command_buffer, queue, cu_idx, kernel_params.n_reconfigure_runs,
              kernel_params.asm_inst_runlist[2 * i],
              kernel_params.reconf_data_runlist[i], constants));
      // Dispatch the new kernel.
      const size_t run_idx = 2 * i + 1;
      const iree_hal_amdxdna_u32_list_t* patch_table =
          run_idx < kernel_params.patch_runlist_count
              ? &kernel_params.patch_runlist[run_idx]
              : nullptr;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_direct_command_buffer_normal_run(
                  bindings, command_buffer, queue, cu_idx,
                  kernel_params.asm_inst_runlist[run_idx], patch_table,
                  constants, /*use_single_partial_elf=*/false));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_begin_debug_group(
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

namespace {
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
}  // namespace
