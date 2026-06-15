// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_NATIVE_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_NATIVE_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/api.h"

// Opaque, driver-private native resources. Platform implementations own the
// concrete handles (Linux KMQ fd/ioctl/mmap objects today, another OS binding
// in the future).
struct iree_hal_amdxdna_native_device_t;
struct iree_hal_amdxdna_native_buffer_t;
struct iree_hal_amdxdna_native_context_t;
struct iree_hal_amdxdna_native_queue_t;
struct iree_hal_amdxdna_native_command_t;
struct iree_hal_amdxdna_native_submission_t;

// Thread-safety contract (current single-worker baseline; the async evolution
// is tracked in the native-DDI follow-ups doc):
//   * Native device objects may be shared across HAL threads.
//   * Context creation and cache lookup are synchronized by the owning HAL
//     device. A native context/queue is a single submission lane unless caps
//     advertise real multi-queue or async submit.
//   * Native command objects are mutable while being built (`add_arg_*`,
//     `bind_buffer`, `reset_bound_buffers`, `mark_*_dirty`, `prepare_chain`);
//     once prepared they are submitted by one lane at a time. Build / rebind /
//     dirty / prepare are preparation-time operations and must not race a
//     native submit or wait on the same command.
//   * Windows MCDM command-aperture staging is queue/context-owned mutable
//     state; submits that touch the same context aperture must stay serialized
//     until an XRT-style reusable instruction lifecycle is recovered.

enum class iree_hal_amdxdna_native_power_mode_t : uint8_t {
  default_mode = 0,
  low,
  medium,
  high,
  turbo,
};

enum class iree_hal_amdxdna_native_buffer_type_t : uint8_t {
  host_only = 0,
  cacheable,
  instruction,
};

enum class iree_hal_amdxdna_native_sync_direction_t : uint8_t {
  host_to_device = 0,
  device_to_host,
};

enum class iree_hal_amdxdna_native_command_opcode_t : uint8_t {
  start_cu = 0,
  start_npu,
  start_npu_partial_elf,
  command_chain,
};

enum iree_hal_amdxdna_native_context_image_model_bits_t : uint32_t {
  IREE_HAL_AMDXDNA_NATIVE_CONTEXT_IMAGE_MODEL_PDI = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_CONTEXT_IMAGE_MODEL_XCLBIN = 1u << 1,
};

enum iree_hal_amdxdna_native_dispatch_model_bits_t : uint32_t {
  IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_CU = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU = 1u << 1,
  IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_PARTIAL_ELF = 1u << 2,
  IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_COMMAND_CHAIN = 1u << 3,
};

enum iree_hal_amdxdna_native_completion_model_bits_t : uint32_t {
  IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_SYNCHRONOUS_WAIT = 1u << 0,
  IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_NATIVE_FENCE = 1u << 1,
  IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_PROGRESS_FENCE = 1u << 2,
  IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_COMPLETION_SLOT = 1u << 3,
};

enum class iree_hal_amdxdna_native_buffer_sync_model_t : uint8_t {
  caller_syncs_bindings = 0,
  submit_syncs_bindings,
};

struct iree_hal_amdxdna_native_device_caps_t {
  uint32_t ddi_version = 1;
  uint32_t max_effective_queues = 1;
  uint32_t max_command_chain_slots = 0;
  uint32_t context_image_models = 0;
  uint32_t dispatch_models = 0;
  iree_hal_amdxdna_native_buffer_sync_model_t buffer_sync_model =
      iree_hal_amdxdna_native_buffer_sync_model_t::caller_syncs_bindings;
  uint32_t completion_models = 0;
  bool supports_command_chain = false;
  bool supports_submit_many = false;
  bool supports_async_submit = false;
  bool supports_external_buffer_import = false;
  bool supports_external_buffer_export = false;
  bool supports_real_multi_queue = false;
  iree_hal_amdxdna_native_command_opcode_t default_dispatch_opcode =
      iree_hal_amdxdna_native_command_opcode_t::start_cu;
};

enum class iree_hal_amdxdna_native_context_image_type_t : uint8_t {
  pdi = 0,
  xclbin,
};

// A context image may carry more than one representation of the same context:
// `type` is the primary format a backend must consume (Linux KMQ requires
// `pdi`; Windows MCDM requires `xclbin`), but both spans can be populated at
// once. The XADX path deliberately provides both: Windows builds the context
// blob from the `xclbin` (AXLF) AND uses the extracted `pdi` for the command
// aperture. This is why the DDI is not a single opaque image blob.
struct iree_hal_amdxdna_native_context_image_t {
  iree_hal_amdxdna_native_context_image_type_t type =
      iree_hal_amdxdna_native_context_image_type_t::pdi;
  iree_const_byte_span_t pdi = {nullptr, 0};
  iree_const_byte_span_t xclbin = {nullptr, 0};
  iree_string_view_t kernel_name = {nullptr, 0};
};

struct iree_hal_amdxdna_native_cu_index_t {
  uint32_t index = 0;
};

struct iree_hal_amdxdna_native_buffer_deleter_t {
  void operator()(iree_hal_amdxdna_native_buffer_t* buffer) const;
};
using iree_hal_amdxdna_native_buffer_ptr =
    std::unique_ptr<iree_hal_amdxdna_native_buffer_t,
                    iree_hal_amdxdna_native_buffer_deleter_t>;

struct iree_hal_amdxdna_native_command_deleter_t {
  void operator()(iree_hal_amdxdna_native_command_t* command) const;
};
using iree_hal_amdxdna_native_command_ptr =
    std::unique_ptr<iree_hal_amdxdna_native_command_t,
                    iree_hal_amdxdna_native_command_deleter_t>;

iree_status_t iree_hal_amdxdna_native_resolve_device_options(
    const iree_hal_amdxdna_device_params* options,
    iree_hal_amdxdna_device_params* out_options,
    std::string* out_device_path_storage,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode);

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device);

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device);

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_power_mode_t power_mode);

iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_device_caps_t* out_caps);

bool iree_hal_amdxdna_native_device_supports_partial_elf_dispatch(
    iree_hal_amdxdna_native_device_t* device);

bool iree_hal_amdxdna_native_device_uses_npu_payload_dispatch(
    iree_hal_amdxdna_native_device_t* device);

bool iree_hal_amdxdna_native_device_syncs_bindings_on_submit(
    iree_hal_amdxdna_native_device_t* device);

iree_hal_amdxdna_native_command_opcode_t
iree_hal_amdxdna_native_device_dispatch_opcode(
    iree_hal_amdxdna_native_device_t* device);

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_type_t type,
    iree_hal_amdxdna_native_buffer_ptr* out_buffer);

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_context_image_t* image,
    iree_hal_amdxdna_native_context_t** out_context);

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context);

iree_status_t iree_hal_amdxdna_native_context_close_single_aperture_session(
    iree_hal_amdxdna_native_context_t* context);

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots);

size_t iree_hal_amdxdna_native_command_arg_binding_capacity();

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr);

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction, iree_device_size_t size,
    iree_device_size_t offset);

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction);

iree_status_t iree_hal_amdxdna_native_buffer_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer);

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_cu_index_t* out_cu_index);

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context);

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue);

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_ptr* out_command);

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_cu_index_t cu_index);

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size);

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value);

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value);

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset);

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size);

// Clears a command's bound buffers so it can be rebound for reuse. Capability
// gated: only callable on backends that advertise the PARTIAL_ELF dispatch
// model (BO-table rebinding). Backends without it may return UNIMPLEMENTED, so
// callers must not invoke it unless `dispatch_models & PARTIAL_ELF` is set.
iree_status_t iree_hal_amdxdna_native_command_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_mark_chain_dirty(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command);

// Builds an ERT_CMD_CHAIN packet from `commands`.
//
// The chain packet copies each child command's exec-BO handle, but does not
// retain the child command objects or their BOs. Callers must keep every child
// command alive until the prepared chain has been submitted and waited.
iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count);

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label);

iree_status_t iree_hal_amdxdna_native_queue_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label);

// Asynchronous submit (advertised via caps.supports_async_submit).
//
// iree_hal_amdxdna_native_queue_submit issues `command` to the hardware queue
// inline on the calling thread (the same path-B work submit_and_wait performs,
// up to and including the SubmitCommandToHwQueue progress-fence kick) and
// returns immediately with an owned submission handle; the hardware is now
// executing. Completion is the hardware progress fence -- there is NO
// background thread.
//
// Serialization: the command aperture is per-context, so the caller MUST keep
// at most one outstanding submission per context/queue and MUST NOT issue again
// on that queue before waiting the previous submission. Distinct contexts/
// queues may overlap freely. (The HAL drives one submission lane per queue,
// which satisfies this.) This DDI does no internal locking.
//
// The caller MUST keep `command` and its bound buffers alive until the
// submission completes. Wait for completion (and obtain the submit status) with
// iree_hal_amdxdna_native_submission_wait, then release the handle with
// iree_hal_amdxdna_native_submission_destroy. submit_and_wait above is exactly
// submit + wait and remains the synchronous path.
iree_status_t iree_hal_amdxdna_native_queue_submit(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label,
    iree_hal_amdxdna_native_submission_t** out_submission);

// Blocks until the submission's hardware fence signals, then performs the
// post-dispatch aperture and device->host output sync and returns the submit
// status. Calling it again after completion returns the cached status.
// NOTE: timeout_ns is not yet honored -- the wait currently runs to completion
// regardless of the deadline (tracked in the native-DDI follow-ups). Pass
// UINT64_MAX for the only currently-meaningful value.
iree_status_t iree_hal_amdxdna_native_submission_wait(
    iree_hal_amdxdna_native_submission_t* submission, uint64_t timeout_ns);

// Non-blocking completion poll: sets *out_ready to true once the hardware
// progress fence has reached this submission's fence value. A ready submission
// still requires submission_wait to retire it (the device->host output sync
// happens in wait), but query lets a caller overlap host work without blocking.
iree_status_t iree_hal_amdxdna_native_submission_query(
    iree_hal_amdxdna_native_submission_t* submission, bool* out_ready);

// Releases a submission handle. If the submission was issued but never waited,
// this waits first so the hardware has finished touching the command and
// aperture before they may be freed.
void iree_hal_amdxdna_native_submission_destroy(
    iree_hal_amdxdna_native_submission_t* submission);

// Diagnostic profiling hook used by benchmark tools. When enabled, native
// implementations may collect submit subphase timings and print them with
// iree_hal_amdxdna_native_debug_dump_submit_profile. Default runtime paths
// remain silent unless a tool explicitly enables this.
void iree_hal_amdxdna_native_debug_set_submit_profile_enabled(bool enabled);

void iree_hal_amdxdna_native_debug_reset_submit_profile();

void iree_hal_amdxdna_native_debug_dump_submit_profile();

#endif  // IREE_HAL_DRIVERS_AMDXDNA_NATIVE_INTERNAL_H_
