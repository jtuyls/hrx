// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/hal/drivers/amdxdna/shim/ert.h"
#include "iree/hal/drivers/amdxdna/shim/windows/mcdm/context_blob.h"
#include "iree/hal/drivers/amdxdna/shim/windows/mcdm/kmt_api.h"

namespace mcdm = iree::hal::amdxdna::mcdm;

namespace {

constexpr uint64_t kMaxExecBoSize = 4096;
// The working Windows/XRT IREE matmul capture submits START_CU/type CU with
// payload count 0x12: one CU mask plus 17 data words. The XML ABI names fewer
// arguments, but XRT pads the tail with zeros and the driver receives 0x4c
// packet bytes.
constexpr uint32_t kWindowsDpuRegmapWords = 17;
constexpr uint32_t kWindowsDpuInstructionRegWord = 2;
constexpr uint64_t kWindowsDpuInstructionApertureOffset = 0x8000;
// XRT's module-runlist path presents each START_NPU child instruction stream at
// a 0x8000-spaced aperture VA (0x04008000, 0x04010000, ...). The path-B parent
// chain descriptor is accepted for one packed child, but multi-child START_NPU
// chains fail unless we preserve this slot cadence.
constexpr size_t kWindowsDpuChainCodeAlignment = 0x8000;
constexpr uint64_t kWindowsDpuChainDescriptorApertureOffset = 0x10000;
constexpr size_t kWindowsDpuChainDescriptorHeaderSize = 0x34;
constexpr size_t kWindowsDpuStartNpuChainDescriptorSize = 0x3c;
// XRT runlists are logically unbounded but internally submitted in fixed-size
// ERT_CMD_CHAIN chunks. XRT 2.19 hardwires that native submit chunk size to 24,
// so the Windows MCDM shim uses the same value for now. The recovered path-B
// descriptor envelope has observed headroom up to 34 children, but 34 is not
// the default until we can prove it is compatible with XRT's intended contract.
// This is a per-native-submit chunk size, not a logical command-chain limit;
// larger logical chains are split by direct_command_buffer.cc before reaching
// this layer.
constexpr size_t kWindowsDpuRunlistSubmitSize = 24;
constexpr uint64_t kWindowsDpuPathBExecBoSize = 0xe0;

struct BoundBuffer {
  size_t position = 0;
  iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
  iree_device_size_t offset = 0;
  iree_device_size_t size = 0;
};

std::string string_view_to_string(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string normalize_cu_name(std::string name) {
  size_t instance_separator = name.find(':');
  if (instance_separator != std::string::npos) {
    name.resize(instance_separator);
  }
  return name;
}

uint64_t windows_dpu_pathb_chain_exec_bo_size() {
  return sizeof(ert_packet) + sizeof(ert_cmd_chain_data) +
         kWindowsDpuRunlistSubmitSize * sizeof(uint64_t);
}

uint32_t chain_slot_capacity(size_t exec_bo_size) {
  const size_t header = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
  return exec_bo_size > header
             ? static_cast<uint32_t>((exec_bo_size - header) / sizeof(uint64_t))
             : 1;
}

bool partial_elf_dummy_bos_enabled() { return true; }

bool partial_elf_bo_table_enabled() { return true; }

bool compact_execbuf_enabled() { return true; }

void flush_host_writes_to_mcdm() {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  FlushProcessWriteBuffers();
}

iree_status_t status_from_mcdm_error(const char* label,
                                     const std::string& error) {
  return iree_make_status(IREE_STATUS_INTERNAL, "%s: %s", label, error.c_str());
}

iree_status_t validate_device_size_fits_u64(iree_device_size_t size) {
  if (IREE_UNLIKELY(size > std::numeric_limits<uint64_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native allocation size is too large");
  }
  return iree_ok_status();
}

iree_status_t parse_power_mode(
    iree_string_view_t power_mode,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_should_set_power_mode = false;
  *out_power_mode = iree_hal_amdxdna_native_power_mode_t::default_mode;
  if (iree_string_view_is_empty(power_mode)) return iree_ok_status();

  *out_should_set_power_mode = true;
  if (iree_string_view_equal(power_mode, IREE_SV("default"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::default_mode;
  } else if (iree_string_view_equal(power_mode, IREE_SV("low"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::low;
  } else if (iree_string_view_equal(power_mode, IREE_SV("medium"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::medium;
  } else if (iree_string_view_equal(power_mode, IREE_SV("high"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::high;
  } else if (iree_string_view_equal(power_mode, IREE_SV("turbo"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::turbo;
  } else {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_power_mode' expected to be default | low | "
        "medium | high | turbo but got '%.*s'",
        static_cast<int>(power_mode.size), power_mode.data);
  }
  return iree_ok_status();
}

mcdm::BufferKind to_mcdm_buffer_kind(
    iree_hal_amdxdna_native_buffer_type_t type) {
  switch (type) {
    case iree_hal_amdxdna_native_buffer_type_t::host_only:
      return mcdm::BufferKind::host_only;
    case iree_hal_amdxdna_native_buffer_type_t::cacheable:
    case iree_hal_amdxdna_native_buffer_type_t::instruction:
      return mcdm::BufferKind::cacheable;
  }
  return mcdm::BufferKind::host_only;
}

uint32_t to_ert_opcode(iree_hal_amdxdna_native_command_opcode_t opcode) {
  switch (opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return ERT_START_CU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu:
      // XRT's non-ELF DPU/TXN path submits DPU kernels as START_CU packets
      // with type ERT_CU. The NPU operation selector is arg0 in the xclbin XML
      // register map, not the ERT packet opcode.
      return ERT_START_CU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf:
      return ERT_START_NPU;
    case iree_hal_amdxdna_native_command_opcode_t::command_chain:
      return ERT_CMD_CHAIN;
  }
  return ERT_START_CU;
}

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command);

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command);

uint32_t first_set_bit(uint32_t value) {
  for (uint32_t i = 0; i < 32; ++i) {
    if (value & (uint32_t{1} << i)) return i;
  }
  return 0;
}

size_t align_up_size(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  mcdm::KmtApi api;
  mcdm::Device device;
  bool pathb_context_ready = false;
  std::vector<iree_hal_amdxdna_native_buffer_ptr> partial_elf_dummy_buffers;

  explicit iree_hal_amdxdna_native_device_t(iree_allocator_t host_allocator)
      : host_allocator(host_allocator) {}
};

struct iree_hal_amdxdna_native_buffer_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  mcdm::Buffer buffer;
  iree_hal_amdxdna_native_buffer_type_t type =
      iree_hal_amdxdna_native_buffer_type_t::host_only;
  bool deferred = false;
  std::vector<uint8_t> deferred_storage;

  iree_hal_amdxdna_native_buffer_t(iree_hal_amdxdna_native_device_t* device,
                                   mcdm::Buffer buffer)
      : device(device),
        buffer(buffer),
        type(iree_hal_amdxdna_native_buffer_type_t::host_only) {}

  iree_hal_amdxdna_native_buffer_t(iree_hal_amdxdna_native_device_t* device,
                                   iree_hal_amdxdna_native_buffer_type_t type,
                                   uint64_t size)
      : device(device),
        type(type),
        deferred(true),
        deferred_storage(static_cast<size_t>(size)) {
    buffer.kind = to_mcdm_buffer_kind(type);
    buffer.size = size;
    buffer.cpu_ptr = deferred_storage.data();
  }

  iree_hal_amdxdna_native_buffer_t(iree_hal_amdxdna_native_device_t* device,
                                   mcdm::BufferKind kind, uint64_t size)
      : device(device),
        type(iree_hal_amdxdna_native_buffer_type_t::cacheable),
        deferred(true),
        deferred_storage(static_cast<size_t>(size)) {
    buffer.kind = kind;
    buffer.size = size;
    buffer.cpu_ptr = deferred_storage.data();
  }
};

struct iree_hal_amdxdna_native_queue_t {
  iree_hal_amdxdna_native_context_t* context = nullptr;
  uint64_t exec_command_count = 0;
};

struct iree_hal_amdxdna_native_context_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  mcdm::Context context;
  mcdm::CommandAperture command_aperture;
  bool has_command_aperture = false;
  iree_device_size_t pathb_single_code_staged_size = 0;
  mcdm::ContextBlobInfo info;
  iree_hal_amdxdna_native_queue_t queue;

  iree_hal_amdxdna_native_context_t(iree_hal_amdxdna_native_device_t* device,
                                    mcdm::Context context,
                                    mcdm::CommandAperture command_aperture,
                                    bool has_command_aperture,
                                    mcdm::ContextBlobInfo info)
      : device(device),
        context(context),
        command_aperture(command_aperture),
        has_command_aperture(has_command_aperture),
        info(std::move(info)) {
    queue.context = this;
  }
};

struct iree_hal_amdxdna_native_command_t {
  iree_hal_amdxdna_native_device_t* device = nullptr;
  iree_hal_amdxdna_native_command_opcode_t opcode;
  iree_hal_amdxdna_native_buffer_ptr exec_buffer;
  iree_hal_amdxdna_native_buffer_t* control_buffer = nullptr;
  iree_device_size_t control_buffer_size = 0;
  ert_start_kernel_cmd* start_packet = nullptr;
  size_t command_size = 0;
  uint32_t cached_start_header = 0;
  bool cached_start_header_valid = false;
  uint32_t reg_idx = 0;
  uint32_t arg_count = 0;
  bool windows_dpu_regmap_finalized = false;
  bool pathb_code_staged = false;
  iree_device_size_t pathb_code_staged_size = 0;
  uint64_t pathb_chain_descriptor_gpu_va = 0;
  uint32_t pathb_chain_descriptor_bytes = 0;
  uint32_t pathb_chain_first_child_opcode = 0;
  uint64_t pathb_chain_code_used_size = 0;
  uint64_t pathb_chain_code_aperture_offset = 0;
  uint64_t pathb_chain_descriptor_aperture_offset = 0;
  bool pathb_chain_allow_code_dedup = true;
  bool pathb_chain_prepared_valid = false;
  bool pathb_chain_code_dirty = false;
  bool pathb_chain_descriptor_dirty = false;
  bool pathb_chain_bound_residency_checked = false;
  std::vector<size_t> pathb_chain_child_code_offsets;
  std::vector<iree_hal_amdxdna_native_command_t*> chain_children;
  std::vector<BoundBuffer> bound_buffers;

  iree_hal_amdxdna_native_command_t(
      iree_hal_amdxdna_native_device_t* device,
      iree_hal_amdxdna_native_command_opcode_t opcode,
      iree_hal_amdxdna_native_buffer_ptr exec_buffer)
      : device(device),
        opcode(opcode),
        exec_buffer(std::move(exec_buffer)),
        start_packet(reinterpret_cast<ert_start_kernel_cmd*>(
            this->exec_buffer->buffer.cpu_ptr)),
        command_size(static_cast<size_t>(this->exec_buffer->buffer.size)) {}
};

iree_status_t materialize_deferred_instruction_buffer(
    iree_hal_amdxdna_native_context_t* context,
    iree_hal_amdxdna_native_buffer_t* buffer);

namespace {

bool pathb_stage_code_after_presync(
    iree_hal_amdxdna_native_command_t* command) {
  return command && command->device;
}

ert_start_kernel_cmd* command_start_packet(
    iree_hal_amdxdna_native_command_t* command) {
  return command->start_packet;
}

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command) {
  return reinterpret_cast<ert_packet*>(command_start_packet(command));
}

void reset_command_packet_for_start(
    iree_hal_amdxdna_native_command_t* command) {
  ert_packet* packet = command_packet(command);
  if (!command->cached_start_header_valid) {
    packet->state = ERT_CMD_STATE_NEW;
    command->cached_start_header = packet->header;
    command->cached_start_header_valid = true;
  }
  packet->header = command->cached_start_header;
  packet->state = ERT_CMD_STATE_NEW;
}

iree_status_t stage_windows_dpu_code_buffer(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires the aperture code BO");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU command has no control-code buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control-code size is not word aligned");
  }
  if (IREE_UNLIKELY(command->control_buffer_size > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM DPU control code exceeds aperture code BO");
  }
  const bool control_is_deferred_pathb_instruction =
      command->device &&
      command->control_buffer->type ==
          iree_hal_amdxdna_native_buffer_type_t::instruction &&
      command->control_buffer->deferred;
  if (!control_is_deferred_pathb_instruction) {
    IREE_RETURN_IF_ERROR(materialize_deferred_instruction_buffer(
        queue->context, command->control_buffer));
  }
  if (queue->context->pathb_single_code_staged_size >
      command->control_buffer_size) {
    const size_t stale_tail_offset =
        static_cast<size_t>(command->control_buffer_size);
    const size_t stale_tail_size =
        static_cast<size_t>(queue->context->pathb_single_code_staged_size -
                            command->control_buffer_size);
    std::memset(
        static_cast<uint8_t*>(aperture.code_cpu_ptr) + stale_tail_offset, 0,
        stale_tail_size);
  }
  std::memcpy(aperture.code_cpu_ptr, command->control_buffer->buffer.cpu_ptr,
              static_cast<size_t>(command->control_buffer_size));
  flush_host_writes_to_mcdm();
  std::string error;
  if (!mcdm::SyncCommandApertureCode(
          command->device->api, command->device->device, aperture,
          kWindowsDpuInstructionApertureOffset,
          static_cast<uint64_t>(command->control_buffer_size), &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM aperture code sync failed", error);
  }
  if (!mcdm::RefreshCommandApertureGpuMapping(
          command->device->api, command->device->device, &aperture, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM aperture code relock failed", error);
  }
  command->pathb_code_staged = true;
  command->pathb_code_staged_size = command->control_buffer_size;
  queue->context->pathb_single_code_staged_size = command->control_buffer_size;
  if (command->opcode ==
      iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf) {
    ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
    if (IREE_UNLIKELY(!npu_data)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no NPU data");
    }
    npu_data->instruction_buffer = aperture.code_gpu_va;
    npu_data->instruction_buffer_size =
        static_cast<uint32_t>(command->control_buffer_size);
    npu_data->instruction_prop_count = 0;
  }
  return iree_ok_status();
}

iree_status_t check_pkt_count_capacity(
    iree_hal_amdxdna_native_command_t* command, uint32_t bytes) {
  if (!command || !command->start_packet) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native command is not initialized");
  }
  uint32_t next_count = command->start_packet->count + bytes / sizeof(uint32_t);
  if (command->command_size <
      sizeof(command->start_packet->header) +
          static_cast<size_t>(next_count) * sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna native command packet is full");
  }
  return iree_ok_status();
}

iree_status_t inc_pkt_count(iree_hal_amdxdna_native_command_t* command,
                            uint32_t bytes) {
  IREE_RETURN_IF_ERROR(check_pkt_count_capacity(command, bytes));
  command->start_packet->count += bytes / sizeof(uint32_t);
  return iree_ok_status();
}

void bind_buffer_ref(iree_hal_amdxdna_native_command_t* command,
                     size_t position, iree_hal_amdxdna_native_buffer_t* buffer,
                     iree_device_size_t offset, iree_device_size_t size) {
  if (position == 0 &&
      command->opcode !=
          iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    command->bound_buffers.clear();
  }
  command->bound_buffers.push_back(BoundBuffer{position, buffer, offset, size});
}

bool is_pathb_partial_elf_control_binding(
    iree_hal_amdxdna_native_command_t* command, const BoundBuffer& bound) {
  return command && command->device &&
         command->opcode ==
             iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf &&
         bound.position == 0 && bound.buffer == command->control_buffer;
}

bool uses_windows_dpu_regmap(iree_hal_amdxdna_native_command_t* command) {
  return command &&
         command->opcode == iree_hal_amdxdna_native_command_opcode_t::start_npu;
}

bool uses_partial_elf_npu_packet(iree_hal_amdxdna_native_command_t* command) {
  return command &&
         command->opcode ==
             iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf;
}

iree_status_t ensure_partial_elf_dummy_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_partial_elf_npu_packet(command) ||
      !partial_elf_dummy_bos_enabled()) {
    return iree_ok_status();
  }
  auto& dummy_buffers = command->device->partial_elf_dummy_buffers;
  while (dummy_buffers.size() < 3) {
    iree_hal_amdxdna_native_buffer_ptr dummy;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_alloc_buffer(
        command->device, /*size=*/4,
        iree_hal_amdxdna_native_buffer_type_t::host_only, &dummy));
    void* ptr = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(dummy.get(), &ptr));
    std::memset(ptr, 0, 4);
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        dummy.get(), iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    dummy_buffers.push_back(std::move(dummy));
  }
  return iree_ok_status();
}

iree_status_t maybe_write_partial_elf_bo_table(
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_partial_elf_npu_packet(command) ||
      !partial_elf_bo_table_enabled()) {
    return iree_ok_status();
  }

  // XRT's module-style command BO carries an out-of-packet table of kernel BO
  // GPU VAs after the 32-byte ERT_START_NPU packet. The inline private packet
  // still advertises only the ERT packet bytes, but the miniport can inspect
  // the command BO payload for dependency/binding metadata. Match that table
  // shape: word 11 starts six 64-bit BO VA slots for the DPU memory args.
  constexpr size_t kBoTableWordOffset = 11;
  constexpr size_t kBoTableEntries = 6;
  constexpr size_t kBoTableWords = 2 * kBoTableEntries;
  const size_t table_bytes =
      (kBoTableWordOffset + kBoTableWords) * sizeof(uint32_t);
  if (IREE_UNLIKELY(command->command_size < table_bytes)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM PARTIAL_ELF command BO is too small for the "
        "XRT-style BO table");
  }

  auto* words = reinterpret_cast<uint32_t*>(command->start_packet);
  std::fill(words + kBoTableWordOffset,
            words + kBoTableWordOffset + kBoTableWords, 0);
  IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(command));
  for (const BoundBuffer& bound : command->bound_buffers) {
    if (!bound.buffer || bound.position == 0) continue;
    const size_t table_index = bound.position - 1;
    if (table_index >= kBoTableEntries) continue;
    uint64_t gpu_va =
        iree_hal_amdxdna_native_buffer_device_address(bound.buffer) +
        bound.offset;
    words[kBoTableWordOffset + 2 * table_index] = static_cast<uint32_t>(gpu_va);
    words[kBoTableWordOffset + 2 * table_index + 1] =
        static_cast<uint32_t>(gpu_va >> 32);
  }
  const auto& dummy_buffers = command->device->partial_elf_dummy_buffers;
  for (size_t i = 0; i < dummy_buffers.size() && i < kBoTableEntries - 3; ++i) {
    uint64_t gpu_va =
        iree_hal_amdxdna_native_buffer_device_address(dummy_buffers[i].get());
    const size_t table_index = 3 + i;
    words[kBoTableWordOffset + 2 * table_index] = static_cast<uint32_t>(gpu_va);
    words[kBoTableWordOffset + 2 * table_index + 1] =
        static_cast<uint32_t>(gpu_va >> 32);
  }
  return iree_ok_status();
}

iree_status_t write_windows_dpu_regmap_u32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  if (command->reg_idx >= kWindowsDpuRegmapWords) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna Windows MCDM DPU register map is full");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t write_windows_dpu_regmap_u64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  if (command->reg_idx + 1 >= kWindowsDpuRegmapWords) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna Windows MCDM DPU register map is full");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[command->reg_idx++] = static_cast<uint32_t>(value);
  regmap[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
  command->arg_count++;
  return iree_ok_status();
}

void set_windows_dpu_instruction_arg(iree_hal_amdxdna_native_command_t* command,
                                     uint64_t instruction_va,
                                     uint32_t instruction_words) {
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  regmap[kWindowsDpuInstructionRegWord] = static_cast<uint32_t>(instruction_va);
  regmap[kWindowsDpuInstructionRegWord + 1] =
      static_cast<uint32_t>(instruction_va >> 32);
  regmap[kWindowsDpuInstructionRegWord + 2] = instruction_words;
}

iree_status_t validate_windows_dpu_regmap_inputs(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (IREE_UNLIKELY(!queue || !queue->context ||
                    !queue->context->has_command_aperture)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires a command aperture");
  }
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU qhdl submit requires the aperture code BO");
  }
  if (IREE_UNLIKELY(!command->control_buffer ||
                    !command->control_buffer->buffer.cpu_ptr ||
                    command->control_buffer_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU command has no control-code buffer");
  }
  if (IREE_UNLIKELY(command->control_buffer_size % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM DPU control-code size is not word aligned");
  }
  if (IREE_UNLIKELY(command->control_buffer_size > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM DPU control code exceeds aperture code BO");
  }
  // The Windows wrapper DPU ABI carries the TXN selector plus a staged
  // instruction pointer. Execute TXNs add three data buffer VAs (arg_count=4,
  // reg_idx=8 before rewrite). Control-packet reconfiguration TXNs add only the
  // control-packet sequence/MC buffer VA (arg_count=2, reg_idx=4); the common
  // rewrite below maps that single VA into the first data slot and leaves the
  // others zero.
  const bool has_execute_args =
      command->arg_count >= 4 && command->reg_idx >= 8;
  const bool has_reconfigure_args =
      command->arg_count == 2 && command->reg_idx == 4;
  if (IREE_UNLIKELY(!has_execute_args && !has_reconfigure_args)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM DPU packet is missing selector/data args");
  }
  return iree_ok_status();
}

iree_status_t rewrite_windows_dpu_regmap_to_instruction(
    iree_hal_amdxdna_native_command_t* command, uint64_t instruction_va) {
  if (command->windows_dpu_regmap_finalized) return iree_ok_status();
  if (IREE_UNLIKELY(command->control_buffer_size / sizeof(uint32_t) >
                    std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "amdxdna Windows MCDM DPU instruction word count is too large");
  }
  uint32_t* regmap = get_ert_regmap_begin(command->start_packet);
  const uint64_t selector = static_cast<uint64_t>(regmap[0]) |
                            (static_cast<uint64_t>(regmap[1]) << 32);
  const uint64_t ifm_va = static_cast<uint64_t>(regmap[2]) |
                          (static_cast<uint64_t>(regmap[3]) << 32);
  const uint64_t param_va = static_cast<uint64_t>(regmap[4]) |
                            (static_cast<uint64_t>(regmap[5]) << 32);
  const uint64_t ofm_va = static_cast<uint64_t>(regmap[6]) |
                          (static_cast<uint64_t>(regmap[7]) << 32);

  std::memset(regmap, 0, kWindowsDpuRegmapWords * sizeof(uint32_t));
  regmap[0] = static_cast<uint32_t>(selector);
  regmap[1] = static_cast<uint32_t>(selector >> 32);
  const uint32_t instruction_words =
      static_cast<uint32_t>(command->control_buffer_size / sizeof(uint32_t));
  set_windows_dpu_instruction_arg(command, instruction_va, instruction_words);
  regmap[5] = static_cast<uint32_t>(ifm_va);
  regmap[6] = static_cast<uint32_t>(ifm_va >> 32);
  regmap[7] = static_cast<uint32_t>(param_va);
  regmap[8] = static_cast<uint32_t>(param_va >> 32);
  regmap[9] = static_cast<uint32_t>(ofm_va);
  regmap[10] = static_cast<uint32_t>(ofm_va >> 32);
  command->reg_idx = kWindowsDpuRegmapWords;
  command->windows_dpu_regmap_finalized = true;
  return iree_ok_status();
}

iree_status_t finalize_windows_dpu_regmap(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command) {
  if (!uses_windows_dpu_regmap(command)) return iree_ok_status();
  if (command->windows_dpu_regmap_finalized) return iree_ok_status();
  IREE_RETURN_IF_ERROR(validate_windows_dpu_regmap_inputs(queue, command));
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  uint64_t instruction_va = aperture.code_gpu_va;

  if (!pathb_stage_code_after_presync(command)) {
    IREE_RETURN_IF_ERROR(stage_windows_dpu_code_buffer(queue, command));
  } else if (command->control_buffer &&
             command->control_buffer->type ==
                 iree_hal_amdxdna_native_buffer_type_t::instruction) {
    IREE_RETURN_IF_ERROR(materialize_deferred_instruction_buffer(
        queue->context, command->control_buffer));
    instruction_va =
        iree_hal_amdxdna_native_buffer_device_address(command->control_buffer);
  }

  IREE_RETURN_IF_ERROR(
      rewrite_windows_dpu_regmap_to_instruction(command, instruction_va));
  return iree_ok_status();
}

iree_status_t append_pathb_start_cu_chain_descriptor(
    iree_hal_amdxdna_native_command_t* child, uint8_t* descriptor_base,
    size_t descriptor_capacity, size_t* descriptor_used) {
  ert_start_kernel_cmd* start = command_start_packet(child);
  if (IREE_UNLIKELY(start->opcode != ERT_START_CU)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM path-B chain descriptor only supports "
        "START_CU children");
  }

  const uint32_t extra_cu_masks = start->extra_cu_masks;
  const uint32_t packet_words = 1u + start->count;
  const uint32_t copy_start_word = 2u + extra_cu_masks;
  if (IREE_UNLIKELY(packet_words < copy_start_word)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B chain child START_CU packet is too short");
  }

  const uint32_t* words = reinterpret_cast<const uint32_t*>(start);
  uint32_t cu_index = 0;
  bool found_cu = false;
  for (uint32_t mask_index = 0; mask_index <= extra_cu_masks; ++mask_index) {
    const uint32_t mask = words[1 + mask_index];
    if (!mask) continue;
    cu_index = mask_index * 32u + first_set_bit(mask);
    found_cu = true;
    break;
  }
  if (IREE_UNLIKELY(!found_cu)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B chain child START_CU packet has no CU "
        "mask bit set");
  }

  const uint32_t copy_words = packet_words - copy_start_word;
  const size_t copy_bytes = static_cast<size_t>(copy_words) * sizeof(uint32_t);
  const size_t descriptor_bytes =
      kWindowsDpuChainDescriptorHeaderSize + copy_bytes;
  if (IREE_UNLIKELY(*descriptor_used > descriptor_capacity ||
                    descriptor_bytes >
                        descriptor_capacity - *descriptor_used)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain descriptor block exceeds 0x%zx "
        "bytes",
        descriptor_capacity);
  }

  uint8_t* descriptor = descriptor_base + *descriptor_used;
  std::memset(descriptor, 0, descriptor_bytes);
  uint32_t value = 1;
  std::memcpy(descriptor + 0x00, &value, sizeof(value));
  value = cu_index;
  std::memcpy(descriptor + 0x2c, &value, sizeof(value));
  value = copy_words;
  std::memcpy(descriptor + 0x30, &value, sizeof(value));
  std::memcpy(descriptor + kWindowsDpuChainDescriptorHeaderSize,
              words + copy_start_word, copy_bytes);
  *descriptor_used += descriptor_bytes;
  return iree_ok_status();
}

iree_status_t append_pathb_start_npu_chain_descriptor(
    iree_hal_amdxdna_native_command_t* child, uint8_t* descriptor_base,
    size_t descriptor_capacity, size_t* descriptor_used) {
  ert_start_kernel_cmd* start = command_start_packet(child);
  if (IREE_UNLIKELY(start->opcode != ERT_START_NPU)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna Windows MCDM path-B START_NPU chain descriptor received a "
        "non-START_NPU child");
  }
  ert_npu_data* npu_data = get_ert_npu_data(start);
  if (IREE_UNLIKELY(!npu_data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM path-B START_NPU chain child has no NPU data");
  }
  if (IREE_UNLIKELY(*descriptor_used > descriptor_capacity ||
                    kWindowsDpuStartNpuChainDescriptorSize >
                        descriptor_capacity - *descriptor_used)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B START_NPU chain descriptor block exceeds "
        "0x%zx bytes",
        descriptor_capacity);
  }

  const uint32_t* args = get_ert_regmap_begin(start);
  const uint32_t* args_end = get_ert_regmap_end(start);
  const ptrdiff_t arg_words = args_end >= args ? args_end - args : 0;
  const uint32_t selector = arg_words > 0 ? args[0] : 0;
  const uint32_t selector_hi = arg_words > 1 ? args[1] : 0;

  uint32_t words[kWindowsDpuStartNpuChainDescriptorSize / sizeof(uint32_t)] =
      {};
  words[0] = 2;
  words[1] = static_cast<uint32_t>(npu_data->instruction_buffer);
  words[2] = static_cast<uint32_t>(npu_data->instruction_buffer >> 32);
  words[7] = npu_data->instruction_buffer_size;
  words[12] = 2;
  words[13] = selector;
  words[14] = selector_hi;

  std::memcpy(descriptor_base + *descriptor_used, words, sizeof(words));
  *descriptor_used += sizeof(words);
  return iree_ok_status();
}

iree_status_t get_pathb_chain_region_sizes(
    iree_hal_amdxdna_native_command_t* chain_command, size_t* out_code_bytes,
    size_t* out_descriptor_bytes) {
  size_t code_offset = 0;
  size_t descriptor_bytes = 0;
  for (iree_hal_amdxdna_native_command_t* child :
       chain_command->chain_children) {
    code_offset = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
    code_offset += static_cast<size_t>(child->control_buffer_size);

    ert_start_kernel_cmd* start = command_start_packet(child);
    if (start->opcode == ERT_START_NPU) {
      descriptor_bytes += kWindowsDpuStartNpuChainDescriptorSize;
      continue;
    }
    if (IREE_UNLIKELY(start->opcode != ERT_START_CU)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains only support START_CU or "
          "START_NPU children");
    }
    const uint32_t packet_words = 1u + start->count;
    const uint32_t copy_start_word = 2u + start->extra_cu_masks;
    if (IREE_UNLIKELY(packet_words < copy_start_word)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child START_CU packet is too "
          "short");
    }
    descriptor_bytes +=
        kWindowsDpuChainDescriptorHeaderSize +
        static_cast<size_t>(packet_words - copy_start_word) * sizeof(uint32_t);
  }
  // XRT's module-runlist aperture layout treats each child instruction stream
  // as occupying a full 0x8000 slot. Reserve through the final slot boundary so
  // batched parent chunks start on the same cadence and opcode-9 marker offsets
  // include the last child slot.
  *out_code_bytes = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
  *out_descriptor_bytes = descriptor_bytes;
  return iree_ok_status();
}

iree_status_t prepare_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* chain_command, bool sync_aperture) {
  if (IREE_UNLIKELY(!queue || !queue->context ||
                    !queue->context->has_command_aperture)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires a command aperture");
  }
  if (IREE_UNLIKELY(chain_command->chain_children.empty())) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "amdxdna Windows MCDM command chain has no child commands");
  }

  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  if (IREE_UNLIKELY(!aperture.code_cpu_ptr || !aperture.code_gpu_va ||
                    aperture.code_size == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires an aperture code BO");
  }
  if (IREE_UNLIKELY(!aperture.gpu_cpu_ptr ||
                    aperture.gpu_va_size <=
                        kWindowsDpuChainDescriptorApertureOffset)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM path-B chain requires the locked command "
        "aperture GPU view");
  }

  if (!sync_aperture && chain_command->pathb_chain_prepared_valid) {
    if (chain_command->pathb_chain_code_dirty) {
      if (IREE_UNLIKELY(chain_command->pathb_chain_child_code_offsets.size() !=
                        chain_command->chain_children.size())) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "amdxdna Windows MCDM path-B prepared chain is missing child code "
            "offsets");
      }
      const uint64_t code_base_offset =
          chain_command->pathb_chain_code_aperture_offset;
      if (IREE_UNLIKELY(code_base_offset >= aperture.code_size)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "amdxdna Windows MCDM prepared path-B chain "
                                "code base offset %" PRIu64
                                " exceeds aperture code BO (%" PRIu64 " bytes)",
                                code_base_offset, aperture.code_size);
      }
      uint8_t* code = static_cast<uint8_t*>(aperture.code_cpu_ptr) +
                      static_cast<size_t>(code_base_offset);
      const size_t code_capacity =
          static_cast<size_t>(aperture.code_size - code_base_offset);
      for (size_t child_index = 0;
           child_index < chain_command->chain_children.size(); ++child_index) {
        iree_hal_amdxdna_native_command_t* child =
            chain_command->chain_children[child_index];
        ert_packet* child_packet = command_packet(child);
        child_packet->state = ERT_CMD_STATE_NEW;
        const size_t code_offset =
            chain_command->pathb_chain_child_code_offsets[child_index];
        const size_t child_code_size =
            static_cast<size_t>(child->control_buffer_size);
        if (IREE_UNLIKELY(code_offset > code_capacity ||
                          child_code_size > code_capacity - code_offset)) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "amdxdna Windows MCDM prepared path-B chain control code "
              "exceeds aperture code BO (%zu-byte child at offset %zu, "
              "capacity %zu)",
              child_code_size, code_offset, code_capacity);
        }
        if (IREE_UNLIKELY(!child->control_buffer ||
                          !child->control_buffer->buffer.cpu_ptr ||
                          child_code_size == 0)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "amdxdna Windows MCDM prepared path-B chain child has no "
              "control-code buffer");
        }
        std::memcpy(code + code_offset, child->control_buffer->buffer.cpu_ptr,
                    child_code_size);
        if (uses_partial_elf_npu_packet(child)) {
          ert_npu_data* npu_data =
              get_ert_npu_data(command_start_packet(child));
          if (IREE_UNLIKELY(!npu_data)) {
            return iree_make_status(
                IREE_STATUS_INTERNAL,
                "amdxdna Windows MCDM PARTIAL_ELF prepared chain child has "
                "no NPU data");
          }
          npu_data->instruction_buffer =
              aperture.code_gpu_va + code_base_offset + code_offset;
          npu_data->instruction_buffer_size =
              static_cast<uint32_t>(child->control_buffer_size);
          npu_data->instruction_prop_count = 0;
          IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(child));
          IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(child));
        }
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            child->exec_buffer.get(),
            iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      }
      chain_command->pathb_chain_descriptor_dirty = false;
      return iree_ok_status();
    }
    for (iree_hal_amdxdna_native_command_t* child :
         chain_command->chain_children) {
      command_packet(child)->state = ERT_CMD_STATE_NEW;
    }
    // Clean prepared chains only reset child ERT state before resubmission.
    // Avoid issuing one KMT sync per child BO on the hot path; the subsequent
    // submit-side host fence makes those state writes visible, while dirty
    // packet/BO-table rewrites still take the explicit sync path above.
    chain_command->pathb_chain_descriptor_dirty = false;
    return iree_ok_status();
  }

  const uint64_t code_base_offset =
      chain_command->pathb_chain_code_aperture_offset;
  if (IREE_UNLIKELY(code_base_offset >= aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain code base offset %" PRIu64
        " exceeds aperture code BO (%" PRIu64 " bytes)",
        code_base_offset, aperture.code_size);
  }
  uint8_t* code = static_cast<uint8_t*>(aperture.code_cpu_ptr) +
                  static_cast<size_t>(code_base_offset);
  const size_t code_capacity =
      static_cast<size_t>(aperture.code_size - code_base_offset);
  size_t code_offset = 0;
  size_t code_used = 0;
  std::vector<size_t> child_code_offsets;
  std::vector<bool> child_code_needs_copy;
  struct UniqueChainCodeSlot {
    iree_hal_amdxdna_native_command_t* child = nullptr;
    size_t offset = 0;
  };
  std::vector<UniqueChainCodeSlot> unique_code_slots;
  child_code_offsets.reserve(chain_command->chain_children.size());
  child_code_needs_copy.reserve(chain_command->chain_children.size());
  unique_code_slots.reserve(chain_command->chain_children.size());
  const bool allow_code_dedup = chain_command->pathb_chain_allow_code_dedup;
  for (iree_hal_amdxdna_native_command_t* child :
       chain_command->chain_children) {
    const size_t child_code_size =
        static_cast<size_t>(child->control_buffer_size);
    bool found_duplicate_code = false;
    if (allow_code_dedup) {
      for (const UniqueChainCodeSlot& slot : unique_code_slots) {
        iree_hal_amdxdna_native_command_t* previous = slot.child;
        if (!previous ||
            previous->control_buffer_size != child->control_buffer_size) {
          continue;
        }
        if (std::memcmp(previous->control_buffer->buffer.cpu_ptr,
                        child->control_buffer->buffer.cpu_ptr,
                        child_code_size) != 0) {
          continue;
        }
        child_code_offsets.push_back(slot.offset);
        child_code_needs_copy.push_back(false);
        found_duplicate_code = true;
        break;
      }
    }
    if (found_duplicate_code) continue;

    code_offset = align_up_size(code_offset, kWindowsDpuChainCodeAlignment);
    if (IREE_UNLIKELY(code_offset > code_capacity ||
                      child_code_size > code_capacity - code_offset)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM path-B chain control code exceeds aperture "
          "code BO (%zu-byte child at offset %zu, capacity %zu)",
          child_code_size, code_offset, code_capacity);
    }
    child_code_offsets.push_back(code_offset);
    child_code_needs_copy.push_back(true);
    unique_code_slots.push_back(UniqueChainCodeSlot{child, code_offset});
    code_used = code_offset + child_code_size;
    code_offset = code_used;
  }
  // Do not clear the full padded code range: START_NPU descriptors carry the
  // exact instruction byte count, and XRT leaves the 0x8000-spaced gaps as
  // allocator slack. Clearing those gaps dominated host-side chain prep.

  const size_t descriptor_offset =
      chain_command->pathb_chain_descriptor_aperture_offset
          ? static_cast<size_t>(
                chain_command->pathb_chain_descriptor_aperture_offset)
          : align_up_size(
                std::max<size_t>(
                    static_cast<size_t>(
                        kWindowsDpuChainDescriptorApertureOffset),
                    static_cast<size_t>(kWindowsDpuInstructionApertureOffset +
                                        code_base_offset) +
                        code_used),
                0x1000);
  if (IREE_UNLIKELY(descriptor_offset >= aperture.gpu_va_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B chain descriptor block exceeds command "
        "aperture after %zu bytes of staged code",
        code_used);
  }
  uint8_t* descriptor_base =
      static_cast<uint8_t*>(aperture.gpu_cpu_ptr) + descriptor_offset;
  const size_t descriptor_capacity =
      static_cast<size_t>(aperture.gpu_va_size - descriptor_offset);
  const size_t descriptor_clear_bytes = std::min<size_t>(
      descriptor_capacity,
      chain_command->chain_children.size() *
          std::max<size_t>(kWindowsDpuStartNpuChainDescriptorSize,
                           kWindowsDpuChainDescriptorHeaderSize +
                               kWindowsDpuRegmapWords * sizeof(uint32_t)));
  std::memset(descriptor_base, 0, descriptor_clear_bytes);

  size_t descriptor_used = 0;
  chain_command->pathb_chain_descriptor_gpu_va = 0;
  chain_command->pathb_chain_descriptor_bytes = 0;
  chain_command->pathb_chain_first_child_opcode =
      command_packet(chain_command->chain_children.front())->opcode;
  for (size_t child_index = 0;
       child_index < chain_command->chain_children.size(); ++child_index) {
    iree_hal_amdxdna_native_command_t* child =
        chain_command->chain_children[child_index];
    ert_packet* child_packet = command_packet(child);
    child_packet->state = ERT_CMD_STATE_NEW;
    if (IREE_UNLIKELY(child_packet->opcode !=
                      chain_command->pathb_chain_first_child_opcode)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains do not support mixed child ERT "
          "opcodes yet");
    }
    const bool is_start_cu_child = uses_windows_dpu_regmap(child);
    const bool is_start_npu_child = uses_partial_elf_npu_packet(child);
    if (IREE_UNLIKELY(!is_start_cu_child && !is_start_npu_child)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna Windows MCDM path-B chains only support DPU commands");
    }
    if (is_start_cu_child && child->windows_dpu_regmap_finalized) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child was already finalized");
    }
    if (IREE_UNLIKELY(!child->control_buffer ||
                      !child->control_buffer->buffer.cpu_ptr ||
                      child->control_buffer_size == 0)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM path-B chain child has no control-code "
          "buffer");
    }
    if (IREE_UNLIKELY(child->control_buffer_size % sizeof(uint32_t) != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna Windows MCDM path-B chain child control-code size is not "
          "word aligned");
    }
    code_offset = child_code_offsets[child_index];
    const size_t child_code_size =
        static_cast<size_t>(child->control_buffer_size);
    if (child_code_needs_copy[child_index]) {
      std::memcpy(code + code_offset, child->control_buffer->buffer.cpu_ptr,
                  child_code_size);
    }
    const uint64_t instruction_va =
        aperture.code_gpu_va + code_base_offset + code_offset;
    if (is_start_cu_child) {
      IREE_RETURN_IF_ERROR(validate_windows_dpu_regmap_inputs(queue, child));
      IREE_RETURN_IF_ERROR(
          rewrite_windows_dpu_regmap_to_instruction(child, instruction_va));
      IREE_RETURN_IF_ERROR(append_pathb_start_cu_chain_descriptor(
          child, descriptor_base, descriptor_capacity, &descriptor_used));
    } else {
      ert_npu_data* npu_data = get_ert_npu_data(command_start_packet(child));
      if (IREE_UNLIKELY(!npu_data)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "amdxdna Windows MCDM PARTIAL_ELF chain child has no NPU data");
      }
      npu_data->instruction_buffer = instruction_va;
      npu_data->instruction_buffer_size =
          static_cast<uint32_t>(child->control_buffer_size);
      npu_data->instruction_prop_count = 0;
      IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(child));
      IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(child));
      IREE_RETURN_IF_ERROR(append_pathb_start_npu_chain_descriptor(
          child, descriptor_base, descriptor_capacity, &descriptor_used));
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        child->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  }

  if (sync_aperture) {
    flush_host_writes_to_mcdm();
    std::string error;
    if (!mcdm::SyncCommandApertureCode(
            chain_command->device->api, chain_command->device->device, aperture,
            kWindowsDpuInstructionApertureOffset + code_base_offset,
            static_cast<uint64_t>(code_used), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B chain aperture code sync failed", error);
    }
    if (descriptor_used) {
      if (!mcdm::SyncCommandApertureCode(
              chain_command->device->api, chain_command->device->device,
              aperture, static_cast<uint64_t>(descriptor_offset),
              static_cast<uint64_t>(descriptor_used), &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B chain descriptor sync failed", error);
      }
    }
    if (!mcdm::RefreshCommandApertureGpuMapping(chain_command->device->api,
                                                chain_command->device->device,
                                                &aperture, &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B chain aperture code relock failed",
          error);
    }
  }
  chain_command->pathb_chain_descriptor_gpu_va =
      aperture.gpu_va + descriptor_offset;
  chain_command->pathb_chain_descriptor_bytes =
      static_cast<uint32_t>(descriptor_used);
  chain_command->pathb_chain_code_used_size =
      allow_code_dedup
          ? code_used
          : align_up_size(code_used, kWindowsDpuChainCodeAlignment);
  chain_command->pathb_chain_child_code_offsets = std::move(child_code_offsets);
  chain_command->pathb_chain_prepared_valid = true;
  chain_command->pathb_chain_code_dirty = true;
  chain_command->pathb_chain_descriptor_dirty = true;
  return iree_ok_status();
}

iree_status_t prepare_pathb_chain_code(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* chain_command) {
  return prepare_pathb_chain_code(queue, chain_command, true);
}

iree_status_t sync_prepared_pathb_chain_batch(
    iree_hal_amdxdna_native_queue_t* queue, iree_host_size_t command_count,
    size_t code_bytes, size_t descriptor_offset, size_t descriptor_bytes) {
  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  flush_host_writes_to_mcdm();
  std::string error;
  const bool use_sync9 = command_count > 1;
  if (use_sync9) {
    size_t last_sync_offset = 0;
    auto submit_sync9 = [&](size_t end_offset) -> iree_status_t {
      if (!end_offset || end_offset == last_sync_offset)
        return iree_ok_status();
      if (!mcdm::SubmitPathBApertureSync(queue->context->device->api,
                                         queue->context->device->device,
                                         &queue->context->context, aperture,
                                         static_cast<uint64_t>(end_offset),
                                         /*wait_for_cpu=*/false, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B batch sync9 failed", error);
      }
      last_sync_offset = end_offset;
      return iree_ok_status();
    };
    // XRT's module-runlist path emits opcode-9 markers at instruction-slot
    // boundaries (0x10000, 0x18000, ...), then submits the ERT_CMD_CHAIN
    // parents. It does not emit a marker for the parent descriptor metadata;
    // our descriptor block is still aperture-resident, so keep the conservative
    // invalidate/relock for that region after the instruction markers.
    const size_t code_end = align_up_size(
        static_cast<size_t>(kWindowsDpuInstructionApertureOffset) + code_bytes,
        kWindowsDpuChainCodeAlignment);
    for (size_t sync_offset = kWindowsDpuInstructionApertureOffset +
                              kWindowsDpuChainCodeAlignment;
         sync_offset <= code_end;
         sync_offset += kWindowsDpuChainCodeAlignment) {
      IREE_RETURN_IF_ERROR(submit_sync9(sync_offset));
    }
    if (descriptor_bytes) {
      if (!mcdm::SyncCommandApertureCode(
              queue->context->device->api, queue->context->device->device,
              aperture, static_cast<uint64_t>(descriptor_offset),
              static_cast<uint64_t>(descriptor_bytes), &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B batch descriptor sync failed", error);
      }
      if (!mcdm::RefreshCommandApertureGpuMapping(
              queue->context->device->api, queue->context->device->device,
              &aperture, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM path-B batch aperture relock failed", error);
      }
    }
    return iree_ok_status();
  }
  if (code_bytes) {
    if (!mcdm::SyncCommandApertureCode(
            queue->context->device->api, queue->context->device->device,
            aperture, kWindowsDpuInstructionApertureOffset,
            static_cast<uint64_t>(code_bytes), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch aperture code sync failed", error);
    }
  }
  if (descriptor_bytes) {
    if (!mcdm::SyncCommandApertureCode(
            queue->context->device->api, queue->context->device->device,
            aperture, static_cast<uint64_t>(descriptor_offset),
            static_cast<uint64_t>(descriptor_bytes), &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM path-B batch descriptor sync failed", error);
    }
  }
  if (!mcdm::RefreshCommandApertureGpuMapping(queue->context->device->api,
                                              queue->context->device->device,
                                              &aperture, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM path-B batch aperture relock failed", error);
  }
  return iree_ok_status();
}

}  // namespace

void iree_hal_amdxdna_native_buffer_deleter_t::operator()(
    iree_hal_amdxdna_native_buffer_t* buffer) const {
  iree_hal_amdxdna_native_buffer_destroy(buffer);
}

void iree_hal_amdxdna_native_command_deleter_t::operator()(
    iree_hal_amdxdna_native_command_t* command) const {
  iree_hal_amdxdna_native_command_destroy(command);
}

iree_status_t materialize_deferred_buffer(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer || !buffer->deferred) return iree_ok_status();
  if (IREE_UNLIKELY(!buffer->device->pathb_context_ready)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM deferred BO materialized before pathb context "
        "setup");
  }

  mcdm::Buffer real_buffer;
  std::string error;
  if (!mcdm::CreateBuffer(buffer->device->api, buffer->device->device,
                          buffer->buffer.kind, buffer->buffer.size,
                          &real_buffer, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM deferred BO allocation failed", error);
  }
  if (real_buffer.cpu_ptr && !buffer->deferred_storage.empty()) {
    const uint64_t copy_size = std::min<uint64_t>(
        buffer->buffer.size,
        static_cast<uint64_t>(buffer->deferred_storage.size()));
    std::memcpy(real_buffer.cpu_ptr, buffer->deferred_storage.data(),
                static_cast<size_t>(copy_size));
  }
  buffer->buffer = real_buffer;
  buffer->deferred = false;
  buffer->deferred_storage.clear();
  buffer->deferred_storage.shrink_to_fit();
  return iree_ok_status();
}

iree_status_t materialize_deferred_instruction_buffer(
    iree_hal_amdxdna_native_context_t* context,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  (void)context;
  return materialize_deferred_buffer(buffer);
}

iree_status_t iree_hal_amdxdna_native_resolve_device_options(
    const iree_hal_amdxdna_device_params* options,
    iree_hal_amdxdna_device_params* out_options,
    std::string* out_device_path_storage,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_options = *options;
  out_device_path_storage->clear();
  if (options->n_core_rows < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_rows' expected a non-negative int32_t but "
        "got %d",
        options->n_core_rows);
  }
  if (options->n_core_cols < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_cols' expected a non-negative int32_t but "
        "got %d",
        options->n_core_cols);
  }
  if (!iree_string_view_is_empty(options->device_path) &&
      !iree_string_view_equal(options->device_path, IREE_SV("default")) &&
      !iree_string_view_equal(options->device_path,
                              IREE_SV("amdxdna://default")) &&
      !iree_string_view_equal(options->device_path, IREE_SV("0"))) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "Windows MCDM amdxdna device path '%.*s' is not available; only the "
        "default adapter is currently supported",
        static_cast<int>(options->device_path.size), options->device_path.data);
  }
  return parse_power_mode(options->power_mode, out_power_mode,
                          out_should_set_power_mode);
}

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  iree_hal_amdxdna_native_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*device), reinterpret_cast<void**>(&device)));
  device = new (device) iree_hal_amdxdna_native_device_t(host_allocator);

  std::string error;
  mcdm::Adapter adapter;
  if (!device->api.Load(&error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM KMT API load failed", error);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  // NO XRT warmup: the pure-KMT replay works deterministically without any XRT
  // (3/3 status=0 on clean firmware). A held-open XRT device actually CONFLICTS
  // with our own kick by taking the NPU context. Talk to the driver directly.

  if (!mcdm::FindNpuAdapter(device->api, &adapter, &error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM adapter discovery failed", error);
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }
  if (!mcdm::CreateDevice(device->api, adapter, &device->device, &error)) {
    iree_status_t status = status_from_mcdm_error(
        "amdxdna Windows MCDM device creation failed", error);
    if (adapter.handle) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = adapter.handle;
      device->api.close_adapter(&close);
    }
    device->~iree_hal_amdxdna_native_device_t();
    iree_allocator_free(host_allocator, device);
    return status;
  }

  *out_device = device;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  if (!device) return;
  iree_allocator_t host_allocator = device->host_allocator;
  mcdm::DestroyDevice(device->api, &device->device);
  device->~iree_hal_amdxdna_native_device_t();
  iree_allocator_free(host_allocator, device);
}

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_power_mode_t power_mode) {
  (void)device;
  if (power_mode == iree_hal_amdxdna_native_power_mode_t::default_mode) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna Windows MCDM power-mode control is not implemented");
}

bool iree_hal_amdxdna_native_device_supports_partial_elf_dispatch(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return (caps.dispatch_models &
          IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_PARTIAL_ELF) != 0;
}

bool iree_hal_amdxdna_native_device_uses_npu_payload_dispatch(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return (caps.dispatch_models &
          IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU) != 0;
}

bool iree_hal_amdxdna_native_device_syncs_bindings_on_submit(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return false;
  }
  return caps.buffer_sync_model ==
         iree_hal_amdxdna_native_buffer_sync_model_t::submit_syncs_bindings;
}

iree_hal_amdxdna_native_command_opcode_t
iree_hal_amdxdna_native_device_dispatch_opcode(
    iree_hal_amdxdna_native_device_t* device) {
  iree_hal_amdxdna_native_device_caps_t caps;
  if (!iree_status_is_ok(
          iree_hal_amdxdna_native_device_query_caps(device, &caps))) {
    return iree_hal_amdxdna_native_command_opcode_t::start_npu;
  }
  return caps.default_dispatch_opcode;
}

iree_status_t iree_hal_amdxdna_native_device_query_caps(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_device_caps_t* out_caps) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_caps);
  iree_hal_amdxdna_native_device_caps_t caps;
  caps.ddi_version = 1;
  caps.max_effective_queues = 1;
  const size_t chain_exec_bo_size =
      static_cast<size_t>(windows_dpu_pathb_chain_exec_bo_size());
  caps.max_command_chain_slots = chain_slot_capacity(chain_exec_bo_size);
  caps.context_image_models =
      IREE_HAL_AMDXDNA_NATIVE_CONTEXT_IMAGE_MODEL_XCLBIN;
  caps.dispatch_models = IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_CU |
                         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_START_NPU |
                         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_PARTIAL_ELF |
                         IREE_HAL_AMDXDNA_NATIVE_DISPATCH_MODEL_COMMAND_CHAIN;
  caps.buffer_sync_model =
      iree_hal_amdxdna_native_buffer_sync_model_t::submit_syncs_bindings;
  caps.completion_models =
      IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_SYNCHRONOUS_WAIT |
      IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_PROGRESS_FENCE |
      IREE_HAL_AMDXDNA_NATIVE_COMPLETION_MODEL_COMPLETION_SLOT;
  caps.supports_command_chain = true;
  caps.supports_submit_many = true;
  caps.supports_async_submit = false;
  caps.supports_external_buffer_import = false;
  caps.supports_external_buffer_export = false;
  caps.supports_real_multi_queue = false;
  caps.default_dispatch_opcode =
      iree_hal_amdxdna_native_command_opcode_t::start_npu;
  *out_caps = caps;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_type_t type,
    iree_hal_amdxdna_native_buffer_ptr* out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  out_buffer->reset();
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));

  const bool defer_pathb_alloc =
      type == iree_hal_amdxdna_native_buffer_type_t::cacheable ||
      type == iree_hal_amdxdna_native_buffer_type_t::instruction;
  if (!device->pathb_context_ready || defer_pathb_alloc) {
    out_buffer->reset(new iree_hal_amdxdna_native_buffer_t(
        device, type, static_cast<uint64_t>(size)));
    return iree_ok_status();
  }

  mcdm::Buffer buffer;
  std::string error;
  if (!mcdm::CreateBuffer(device->api, device->device,
                          to_mcdm_buffer_kind(type),
                          static_cast<uint64_t>(size), &buffer, &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM BO allocation failed",
                                  error);
  }
  out_buffer->reset(new iree_hal_amdxdna_native_buffer_t(device, buffer));
  (*out_buffer)->type = type;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device,
    const iree_hal_amdxdna_native_context_image_t* image,
    iree_hal_amdxdna_native_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(image);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = nullptr;
  if (IREE_UNLIKELY(image->type !=
                    iree_hal_amdxdna_native_context_image_type_t::xclbin)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM context creation requires an xclbin context "
        "image; compile with --iree-amdaie-amdxdna-emit-context-xclbin=true");
  }
  iree_const_byte_span_t pdi = image->pdi;
  iree_const_byte_span_t xclbin = image->xclbin;
  if (IREE_UNLIKELY(xclbin.data_length == 0 || !xclbin.data)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM context creation requires xclbin data; "
        "compile with --iree-amdaie-amdxdna-emit-context-xclbin=true");
  }

  std::vector<uint8_t> private_data;
  mcdm::ContextBlobInfo info;
  std::string error;
  if (!mcdm::BuildContextPrivateDataFromXclbin(xclbin.data, xclbin.data_length,
                                               GetCurrentProcessId(),
                                               &private_data, &info, &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context blob generation failed", error);
  }
  mcdm::Context context;
  if (!mcdm::CreateContext(device->api, device->device, private_data, &context,
                           &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM context creation failed", error);
  }
  mcdm::CommandAperture command_aperture = {};
  bool has_command_aperture = false;
  if (!mcdm::CreateCommandAperture(device->api, device->device, context,
                                   &command_aperture, &error)) {
    mcdm::DestroyContext(device->api, &context);
    return status_from_mcdm_error(
        "amdxdna Windows MCDM command aperture creation failed", error);
  }
  has_command_aperture = true;
  context.completion_ring.kind = mcdm::BufferKind::cacheable;
  context.completion_ring.size = command_aperture.allocation_size;
  context.completion_ring.allocation = command_aperture.allocation;
  context.completion_ring.resource = command_aperture.resource;
  context.completion_ring.cpu_ptr = command_aperture.cpu_ptr;
  context.completion_ring_ready = true;
  context.completion_ring_offset = 8;
  if (!mcdm::SubmitAndWaitPathBSetup(device->api, device->device, &context,
                                     &command_aperture, pdi.data,
                                     pdi.data_length, &error)) {
    mcdm::DestroyCommandAperture(device->api, device->device,
                                 &command_aperture);
    mcdm::DestroyContext(device->api, &context);
    return status_from_mcdm_error("amdxdna Windows MCDM pathb setup failed",
                                  error);
  }
  device->pathb_context_ready = true;

  *out_context = new iree_hal_amdxdna_native_context_t(
      device, context, command_aperture, has_command_aperture, info);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_t** out_context) {
  iree_hal_amdxdna_native_context_image_t image;
  image.type = iree_hal_amdxdna_native_context_image_type_t::xclbin;
  image.pdi = pdi;
  image.xclbin = xclbin;
  image.kernel_name = kernel_name;
  return iree_hal_amdxdna_native_device_create_context(device, &image,
                                                       out_context);
}

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context) {
  if (!context) return;
  if (context->has_command_aperture) {
    mcdm::DestroyCommandAperture(context->device->api, context->device->device,
                                 &context->command_aperture);
  }
  mcdm::DestroyContext(context->device->api, &context->context);
  delete context;
}

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_max_slots);
  iree_hal_amdxdna_native_device_caps_t caps;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_device_query_caps(device, &caps));
  *out_max_slots = caps.max_command_chain_slots;
  return iree_ok_status();
}

size_t iree_hal_amdxdna_native_command_arg_binding_capacity() { return 1024; }

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  if (!buffer) return;
  if (!buffer->deferred) {
    mcdm::DestroyBuffer(buffer->device->api, buffer->device->device,
                        &buffer->buffer);
  }
  delete buffer;
}

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = nullptr;
  if (IREE_UNLIKELY(!buffer || !buffer->buffer.cpu_ptr)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not host-mapped");
  }
  *out_ptr = buffer->buffer.cpu_ptr;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction, iree_device_size_t size,
    iree_device_size_t offset) {
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(size));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_u64(offset));
  if (buffer->deferred) {
    return iree_ok_status();
  }
  if (direction == iree_hal_amdxdna_native_sync_direction_t::host_to_device) {
    flush_host_writes_to_mcdm();
    return iree_ok_status();
  }
  std::string error;
  if (!mcdm::SyncBuffer(buffer->device->api, buffer->device->device,
                        buffer->buffer, static_cast<uint64_t>(offset),
                        static_cast<uint64_t>(size), &error)) {
    return status_from_mcdm_error("amdxdna Windows MCDM BO sync failed", error);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction) {
  return iree_hal_amdxdna_native_buffer_sync(
      buffer, direction, iree_hal_amdxdna_native_buffer_size(buffer), 0);
}

iree_status_t iree_hal_amdxdna_native_buffer_ensure_allocated(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return materialize_deferred_buffer(buffer);
}

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return buffer->buffer.gpu_va;
}

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return static_cast<iree_device_size_t>(buffer->buffer.size);
}

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_cu_index_t* out_cu_index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cu_index);

  const std::vector<std::string>& kernel_names = context->info.kernel_names;
  if (kernel_names.empty()) {
    out_cu_index->index = 0;
    return iree_ok_status();
  }

  std::string requested = normalize_cu_name(string_view_to_string(kernel_name));
  for (size_t i = 0; i < kernel_names.size(); ++i) {
    if (requested == kernel_names[i]) {
      out_cu_index->index = static_cast<uint32_t>(i);
      return iree_ok_status();
    }
  }

  if (kernel_names.size() == 1) {
    out_cu_index->index = 0;
    return iree_ok_status();
  }

  std::string available;
  for (size_t i = 0; i < kernel_names.size(); ++i) {
    if (i) available += ", ";
    available += kernel_names[i];
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "amdxdna Windows MCDM context does not contain requested CU '%s'; "
      "available CUs: %s",
      requested.c_str(), available.c_str());
}

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context) {
  return &context->queue;
}

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return queue->exec_command_count;
}

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_ptr* out_command) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command);
  out_command->reset();

  iree_hal_amdxdna_native_buffer_ptr exec_buffer;
  iree_status_t status = iree_ok_status();
  uint64_t exec_buffer_size = kMaxExecBoSize;
  if (opcode == iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    exec_buffer_size = windows_dpu_pathb_chain_exec_bo_size();
  } else if (compact_execbuf_enabled()) {
    exec_buffer_size = kWindowsDpuPathBExecBoSize;
  }
  exec_buffer.reset(new iree_hal_amdxdna_native_buffer_t(
      device, mcdm::BufferKind::execbuf, exec_buffer_size));
  auto* command = new iree_hal_amdxdna_native_command_t(device, opcode,
                                                        std::move(exec_buffer));
  std::memset(command->start_packet, 0, command->command_size);
  command->start_packet->state = ERT_CMD_STATE_NEW;
  command->start_packet->opcode = to_ert_opcode(opcode);
  command->start_packet->type = ERT_CU;
  status = inc_pkt_count(command, sizeof(uint32_t));
  if (!iree_status_is_ok(status)) {
    delete command;
    return status;
  }
  if (opcode == iree_hal_amdxdna_native_command_opcode_t::start_npu) {
    // XRT creates the full DPU register-map payload up front:
    // header + CU mask + 15 register words from the xclbin XML metadata.
    command->start_packet->count = 1 + kWindowsDpuRegmapWords;
  } else if (opcode ==
             iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf) {
    // Match XRT's module-style ERT_START_NPU packet:
    // header + CU mask + ert_npu_data + selector word + trailing return word.
    command->start_packet->count =
        1 + sizeof(ert_npu_data) / sizeof(uint32_t) + 2;
  }
  out_command->reset(command);
  return iree_ok_status();
}

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  delete command;
}

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_cu_index_t cu_index) {
  command->start_packet->cu_mask = 0x1u << cu_index.index;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer,
    iree_device_size_t control_buffer_size) {
  switch (command->opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return iree_ok_status();
    case iree_hal_amdxdna_native_command_opcode_t::start_npu: {
      if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM instruction buffer is empty");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        iree_hal_amdxdna_native_buffer_size(control_buffer))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM instruction byte count exceeds BO size");
      }
      if (IREE_UNLIKELY(control_buffer_size % sizeof(uint32_t) != 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM instruction byte count is not word aligned");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        std::numeric_limits<uint32_t>::max())) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM instruction buffer is too large");
      }
      command->control_buffer = control_buffer;
      command->control_buffer_size = control_buffer_size;
      command->pathb_code_staged = false;
      command->pathb_code_staged_size = 0;
      return iree_ok_status();
    }
    case iree_hal_amdxdna_native_command_opcode_t::start_npu_partial_elf: {
      if (IREE_UNLIKELY(!control_buffer || control_buffer_size == 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM PARTIAL_ELF instruction buffer is empty");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        iree_hal_amdxdna_native_buffer_size(control_buffer))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM PARTIAL_ELF instruction byte count exceeds "
            "BO size");
      }
      if (IREE_UNLIKELY(control_buffer_size % sizeof(uint32_t) != 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna Windows MCDM PARTIAL_ELF instruction byte count is not "
            "word aligned");
      }
      if (IREE_UNLIKELY(control_buffer_size >
                        std::numeric_limits<uint32_t>::max())) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "amdxdna Windows MCDM PARTIAL_ELF instruction buffer is too large");
      }
      command->control_buffer = control_buffer;
      command->control_buffer_size = control_buffer_size;
      command->pathb_code_staged = false;
      command->pathb_code_staged_size = 0;
      ert_npu_data* npu_data = get_ert_npu_data(command->start_packet);
      if (IREE_UNLIKELY(!npu_data)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "amdxdna Windows MCDM PARTIAL_ELF packet has no NPU data");
      }
      // Path-B exposes the instruction stream through the context instruction
      // BO. The control buffer is host-side source data staged there before
      // submit.
      npu_data->instruction_buffer = 0;
      npu_data->instruction_buffer_size =
          static_cast<uint32_t>(control_buffer_size);
      npu_data->instruction_prop_count = 0;
      bind_buffer_ref(command, /*position=*/0, control_buffer, /*offset=*/0,
                      control_buffer_size);
      return iree_ok_status();
    }
    case iree_hal_amdxdna_native_command_opcode_t::command_chain:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported control buffer command opcode");
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  if (uses_windows_dpu_regmap(command)) {
    if (command->arg_count == 0) {
      return write_windows_dpu_regmap_u64(command, value);
    }
    return write_windows_dpu_regmap_u32(command, value);
  }
  if (uses_partial_elf_npu_packet(command)) {
    if (command->reg_idx >= 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no free arg slots");
    }
    uint32_t* args = get_ert_regmap_begin(command->start_packet);
    args[command->reg_idx++] = value;
    command->arg_count++;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = value;
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  if (uses_windows_dpu_regmap(command)) {
    return write_windows_dpu_regmap_u64(command, value);
  }
  if (uses_partial_elf_npu_packet(command)) {
    if (command->reg_idx + 1 >= 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna Windows MCDM PARTIAL_ELF packet has no free u64 arg slot");
    }
    uint32_t* args = get_ert_regmap_begin(command->start_packet);
    args[command->reg_idx++] = static_cast<uint32_t>(value);
    args[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
    command->arg_count++;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(inc_pkt_count(command, sizeof(value)));
  auto args = get_ert_regmap_begin(command->start_packet);
  args[command->reg_idx++] = static_cast<uint32_t>(value);
  args[command->reg_idx++] = static_cast<uint32_t>(value >> 32);
  command->arg_count++;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(command,
                                                                  buffer, 0);
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  IREE_RETURN_IF_ERROR(materialize_deferred_buffer(buffer));
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer offset too large");
  }
  if (uses_partial_elf_npu_packet(command)) {
    bind_buffer_ref(command, command->arg_count, buffer, offset,
                    iree_hal_amdxdna_native_buffer_size(buffer) - offset);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(check_pkt_count_capacity(command, sizeof(uint64_t)));
  bind_buffer_ref(command, command->arg_count, buffer, offset,
                  iree_hal_amdxdna_native_buffer_size(buffer) - offset);
  return iree_hal_amdxdna_native_command_add_arg_64(
      command, iree_hal_amdxdna_native_buffer_device_address(buffer) + offset);
}

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  IREE_RETURN_IF_ERROR(materialize_deferred_buffer(buffer));
  if (offset > iree_hal_amdxdna_native_buffer_size(buffer) ||
      size > iree_hal_amdxdna_native_buffer_size(buffer) - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command buffer binding range is "
                            "out of bounds");
  }
  bind_buffer_ref(command, position, buffer, offset, size);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_reset_bound_buffers(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  command->bound_buffers.clear();
  if (uses_partial_elf_npu_packet(command) && command->control_buffer) {
    bind_buffer_ref(command, /*position=*/0, command->control_buffer,
                    /*offset=*/0, command->control_buffer_size);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  command->pathb_chain_prepared_valid = false;
  command->pathb_chain_code_dirty = true;
  command->pathb_chain_descriptor_dirty = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (command->opcode ==
      iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    command->pathb_chain_code_dirty = true;
    if (!command->pathb_chain_prepared_valid) {
      command->pathb_chain_descriptor_dirty = true;
    }
  } else {
    command->pathb_code_staged = false;
    command->pathb_code_staged_size = 0;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_mark_chain_code_dirty(
    iree_hal_amdxdna_native_command_t* command) {
  IREE_ASSERT_ARGUMENT(command);
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (!command->pathb_chain_prepared_valid) {
    command->pathb_chain_descriptor_dirty = true;
  }
  command->pathb_chain_code_dirty = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (IREE_UNLIKELY(command_count > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command chain is too large");
  }
  const size_t chain_bytes = offsetof(ert_packet, data) +
                             sizeof(ert_cmd_chain_data) +
                             command_count * sizeof(uint64_t);
  if (chain_bytes > command->command_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna cmd-chain: %" PRIhsz
                            " slots exceed exec buffer (%zu > %zu bytes)",
                            command_count, chain_bytes, command->command_size);
  }

  ert_packet* packet = command_packet(command);
  std::memset(packet, 0, command->command_size);
  command->cached_start_header_valid = false;
  command->chain_children.clear();
  command->bound_buffers.clear();
  packet->state = ERT_CMD_STATE_NEW;
  packet->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* chain_data =
      reinterpret_cast<ert_cmd_chain_data*>(packet->data);
  chain_data->command_count = static_cast<uint32_t>(command_count);
  chain_data->submit_index = 0;
  chain_data->error_index = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    IREE_RETURN_IF_ERROR(
        materialize_deferred_buffer(commands[i]->exec_buffer.get()));
    // Materializing a deferred exec BO replaces the temporary host-storage
    // mapping with the real KMT allocation mapping. Refresh the cached packet
    // pointer before any later descriptor construction reads the child.
    commands[i]->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        commands[i]->exec_buffer->buffer.cpu_ptr);
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        commands[i]->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    // XRT's source-level runlist path stores the child run BO's kernel-mode
    // handle here and then calls bind_at() on the parent chain BO.
    chain_data->data[i] = commands[i]->exec_buffer->buffer.allocation;
    command->chain_children.push_back(commands[i]);
    command->bound_buffers.push_back(BoundBuffer{
        i, commands[i]->exec_buffer.get(), 0,
        iree_hal_amdxdna_native_buffer_size(commands[i]->exec_buffer.get())});
  }
  packet->count =
      (sizeof(ert_cmd_chain_data) + command_count * sizeof(uint64_t)) /
      sizeof(uint32_t);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command);
  ert_packet* packet = command_packet(command);
  reset_command_packet_for_start(command);

  IREE_RETURN_IF_ERROR(finalize_windows_dpu_regmap(queue, command));

  std::string error;
  for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
    const BoundBuffer& bound = command->bound_buffers[i];
    if (!bound.buffer) continue;
    if (is_pathb_partial_elf_control_binding(command, bound)) continue;
    IREE_RETURN_IF_ERROR(materialize_deferred_buffer(bound.buffer));
    std::string label = "bound[" + std::to_string(i) + "]";
    if (!mcdm::WaitForBufferResidency(
            command->device->api, command->device->device,
            queue->context->context, bound.buffer->buffer, label.c_str(),
            &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM bound BO residency wait failed", error);
    }
  }

  bool packet_state_from_completion_slot = false;
  {
    const bool is_pathb_chain =
        command->opcode ==
        iree_hal_amdxdna_native_command_opcode_t::command_chain;
    const bool is_pathb_partial_elf =
        !is_pathb_chain && uses_partial_elf_npu_packet(command);
    const uint32_t command_bytes = (packet->count + 1) * sizeof(uint32_t);
    const bool skip_bound_sync = !is_pathb_chain && is_pathb_partial_elf;
    // The NPU is not cache-coherent: flush every bound buffer (instruction
    // control code + input args) host->device BEFORE the dispatch so the
    // firmware reads real data, not stale device memory. (Output is synced
    // device->host after.)
    if (!skip_bound_sync) {
      for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
        iree_hal_amdxdna_native_buffer_t* bound =
            command->bound_buffers[i].buffer;
        if (!bound) continue;
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            bound, iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      }
    }
    if (!queue->context->has_command_aperture) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna Windows MCDM pathb submit requested without command "
          "aperture");
    }
    const bool skip_non_chain_presync = !is_pathb_chain && is_pathb_partial_elf;
    if (!skip_non_chain_presync) {
      if (!mcdm::SubmitPathBApertureSync(
              command->device->api, command->device->device,
              &queue->context->context, queue->context->command_aperture,
              /*offset=*/0x10000, /*wait_for_cpu=*/false, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb pre-dispatch sync failed", error);
      }
    }
    if (is_pathb_chain) {
      IREE_RETURN_IF_ERROR(prepare_pathb_chain_code(queue, command));
    } else if (pathb_stage_code_after_presync(command)) {
      IREE_RETURN_IF_ERROR(stage_windows_dpu_code_buffer(queue, command));
    }
    IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(command));
    IREE_RETURN_IF_ERROR(
        materialize_deferred_buffer(command->exec_buffer.get()));
    command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        command->exec_buffer->buffer.cpu_ptr);
    packet = command_packet(command);
    IREE_RETURN_IF_ERROR(maybe_write_partial_elf_bo_table(command));
    // Module-style path-B writes the state-3 command BO through its CPU
    // mapping and submits it directly. For partial-ELF path-B commands, use a
    // CPU fence instead of a generic per-dispatch host->device sync.
    const bool skip_exec_sync = !is_pathb_chain && is_pathb_partial_elf;
    if (!skip_exec_sync) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
          command->exec_buffer.get(),
          iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    } else {
      std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    if (is_pathb_chain) {
      mcdm::PathBChainSubmitInfo chain_info = {};
      chain_info.descriptor_gpu_va = command->pathb_chain_descriptor_gpu_va;
      chain_info.descriptor_bytes = command->pathb_chain_descriptor_bytes;
      chain_info.command_count =
          reinterpret_cast<ert_cmd_chain_data*>(packet->data)->command_count;
      chain_info.first_child_opcode = command->pathb_chain_first_child_opcode;
      if (!mcdm::SubmitAndWaitPathBChain(
              command->device->api, command->device->device,
              &queue->context->context, command->exec_buffer->buffer, packet,
              command_bytes, chain_info, &packet->header, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb chain submit failed", error);
      }
    } else {
      if (!mcdm::SubmitAndWaitPathB(
              command->device->api, command->device->device,
              &queue->context->context, command->exec_buffer->buffer, packet,
              command_bytes, /*command_state=*/3, &packet->header, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb command submit failed", error);
      }
    }
    packet_state_from_completion_slot = true;
    const bool skip_non_chain_postsync =
        !is_pathb_chain && is_pathb_partial_elf;
    if (!skip_non_chain_postsync) {
      if (!mcdm::SubmitPathBApertureSync(
              command->device->api, command->device->device,
              &queue->context->context, queue->context->command_aperture,
              /*offset=*/0x8000, /*wait_for_cpu=*/true, &error)) {
        return status_from_mcdm_error(
            "amdxdna Windows MCDM pathb post-dispatch sync failed", error);
      }
    }
    // The NPU is not cache-coherent: invalidate every bound buffer (incl. the
    // output) device->host so the host reads the firmware's results, not stale
    // cache. This was missing and could masquerade as "no execution".
    if (!skip_bound_sync) {
      for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
        iree_hal_amdxdna_native_buffer_t* bound =
            command->bound_buffers[i].buffer;
        if (!bound) continue;
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
            bound, iree_hal_amdxdna_native_sync_direction_t::device_to_host));
      }
    }
  }
  queue->exec_command_count++;

  if (!packet_state_from_completion_slot) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        command->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::device_to_host));
  }
  if (packet->state == ERT_CMD_STATE_COMPLETED) return iree_ok_status();

  if (command->opcode ==
      iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s did not complete: ert state %u (error_index %u, "
        "submit_index %u)",
        static_cast<int>(label.size), label.data, packet->state,
        chain_data->error_index, chain_data->submit_index);
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL, "amdxdna %.*s did not complete: ert state %u",
      static_cast<int>(label.size), label.data, packet->state);
}

iree_status_t iree_hal_amdxdna_native_queue_submit_all_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(commands);
  if (command_count == 0) return iree_ok_status();

  iree_hal_amdxdna_native_device_t* device = queue->context->device;

  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (commands[i]->opcode !=
        iree_hal_amdxdna_native_command_opcode_t::command_chain) {
      for (iree_host_size_t j = 0; j < command_count; ++j) {
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_queue_submit_and_wait(
            queue, commands[j], label));
      }
      return iree_ok_status();
    }
  }

  if (!queue->context->has_command_aperture) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna Windows MCDM pathb batch submit requested without command "
        "aperture");
  }

  mcdm::CommandAperture& aperture = queue->context->command_aperture;
  std::vector<size_t> code_sizes(command_count);
  std::vector<size_t> descriptor_sizes(command_count);
  size_t code_cursor = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    // Deduplicating identical child instruction streams within one parent chain
    // is valid and matches the driver's per-descriptor byte-count model. Keep
    // multi-parent batches conservative: submitting multiple parents before a
    // wait failed when they shared compacted aperture slots, so give each child
    // its own XRT-style 0x8000 slot until that firmware ordering rule is
    // mapped.
    commands[i]->pathb_chain_allow_code_dedup = command_count == 1;
    IREE_RETURN_IF_ERROR(get_pathb_chain_region_sizes(
        commands[i], &code_sizes[i], &descriptor_sizes[i]));
    const size_t code_base = align_up_size(code_cursor, 0x1000);
    commands[i]->pathb_chain_code_aperture_offset = code_base;
    code_cursor = code_base + align_up_size(code_sizes[i], 0x1000);
  }
  if (IREE_UNLIKELY(code_cursor > aperture.code_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B batch chain code uses %zu bytes, "
        "exceeding aperture code BO (%" PRIu64 " bytes)",
        code_cursor, aperture.code_size);
  }

  size_t descriptor_cursor = align_up_size(
      std::max<size_t>(
          static_cast<size_t>(kWindowsDpuChainDescriptorApertureOffset),
          static_cast<size_t>(kWindowsDpuInstructionApertureOffset) +
              code_cursor),
      0x1000);
  const size_t descriptor_batch_offset = descriptor_cursor;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    commands[i]->pathb_chain_descriptor_aperture_offset = descriptor_cursor;
    descriptor_cursor += align_up_size(descriptor_sizes[i], 0x1000);
  }
  if (IREE_UNLIKELY(descriptor_cursor > aperture.gpu_va_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna Windows MCDM path-B batch chain descriptors use aperture "
        "through byte %zu, exceeding command aperture (%" PRIu64 " bytes)",
        descriptor_cursor, aperture.gpu_va_size);
  }

  std::vector<mcdm::PathBPendingSubmit> pending(command_count);
  std::string error;
  size_t batch_code_sync_bytes = 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    reset_command_packet_for_start(command);

    IREE_RETURN_IF_ERROR(finalize_windows_dpu_regmap(queue, command));

    if (!command->pathb_chain_bound_residency_checked) {
      for (size_t i = 0; i < command->bound_buffers.size(); ++i) {
        const BoundBuffer& bound = command->bound_buffers[i];
        if (!bound.buffer) continue;
        IREE_RETURN_IF_ERROR(materialize_deferred_buffer(bound.buffer));
        std::string residency_label = "batch-bound[" + std::to_string(i) + "]";
        if (!mcdm::WaitForBufferResidency(
                command->device->api, command->device->device,
                queue->context->context, bound.buffer->buffer,
                residency_label.c_str(), &error)) {
          return status_from_mcdm_error(
              "amdxdna Windows MCDM bound BO residency wait failed", error);
        }
      }
      command->pathb_chain_bound_residency_checked = true;
    }

    IREE_RETURN_IF_ERROR(prepare_pathb_chain_code(queue, command, false));
    if (command->pathb_chain_code_dirty ||
        command->pathb_chain_descriptor_dirty) {
      batch_code_sync_bytes = std::max<size_t>(
          batch_code_sync_bytes,
          static_cast<size_t>(command->pathb_chain_code_aperture_offset +
                              command->pathb_chain_code_used_size));
    }
    IREE_RETURN_IF_ERROR(ensure_partial_elf_dummy_buffers(command));
    IREE_RETURN_IF_ERROR(
        materialize_deferred_buffer(command->exec_buffer.get()));
    command->start_packet = reinterpret_cast<ert_start_kernel_cmd*>(
        command->exec_buffer->buffer.cpu_ptr);
    packet = command_packet(command);
    reset_command_packet_for_start(command);

    // Chain parent bound buffers are the child exec BOs. The path-B chain
    // preparation above rewrites each child packet and syncs each exec BO
    // exactly once; syncing the parent bindings here duplicates that work for
    // every slot in every native parent chunk.
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        command->exec_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  }

  size_t dirty_descriptor_begin = std::numeric_limits<size_t>::max();
  size_t dirty_descriptor_end = 0;
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    if (!command->pathb_chain_descriptor_dirty) continue;
    const size_t begin =
        static_cast<size_t>(command->pathb_chain_descriptor_aperture_offset);
    const size_t end =
        begin + static_cast<size_t>(command->pathb_chain_descriptor_bytes);
    dirty_descriptor_begin = std::min(dirty_descriptor_begin, begin);
    dirty_descriptor_end = std::max(dirty_descriptor_end, end);
  }
  const size_t descriptor_sync_offset =
      dirty_descriptor_begin == std::numeric_limits<size_t>::max()
          ? descriptor_batch_offset
          : dirty_descriptor_begin;
  const size_t descriptor_sync_bytes =
      dirty_descriptor_begin == std::numeric_limits<size_t>::max()
          ? 0
          : dirty_descriptor_end - dirty_descriptor_begin;

  IREE_RETURN_IF_ERROR(sync_prepared_pathb_chain_batch(
      queue, command_count, batch_code_sync_bytes, descriptor_sync_offset,
      descriptor_sync_bytes));
  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    commands[command_index]->pathb_chain_code_dirty = false;
    commands[command_index]->pathb_chain_descriptor_dirty = false;
  }

  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    mcdm::PathBChainSubmitInfo chain_info = {};
    chain_info.descriptor_gpu_va = command->pathb_chain_descriptor_gpu_va;
    chain_info.descriptor_bytes = command->pathb_chain_descriptor_bytes;
    chain_info.command_count =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data)->command_count;
    chain_info.first_child_opcode = command->pathb_chain_first_child_opcode;
    if (!mcdm::SubmitPathBChain(
            command->device->api, command->device->device,
            &queue->context->context, command->exec_buffer->buffer, packet,
            (packet->count + 1) * sizeof(uint32_t), chain_info, &packet->header,
            &pending[command_index], &error)) {
      return status_from_mcdm_error(
          "amdxdna Windows MCDM pathb chain batch submit failed", error);
    }
  }

  if (!mcdm::WaitForPathBSubmits(device->api, device->device,
                                 &queue->context->context, pending.data(),
                                 pending.size(), &error)) {
    return status_from_mcdm_error(
        "amdxdna Windows MCDM pathb chain batch wait failed", error);
  }

  for (iree_host_size_t command_index = 0; command_index < command_count;
       ++command_index) {
    iree_hal_amdxdna_native_command_t* command = commands[command_index];
    ert_packet* packet = command_packet(command);
    // The direct command-buffer chain flush invalidates the exact I/O binding
    // ranges once after the whole group completes. Avoid invalidating every
    // bound BO for every native parent chunk here; that duplicates work and was
    // the dominant batched-chain host overhead.
    queue->exec_command_count++;
    if (packet->state == ERT_CMD_STATE_COMPLETED) continue;
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s batch command %" PRIhsz
        " did not complete: ert state %u (error_index %u, submit_index %u)",
        static_cast<int>(label.size), label.data, command_index, packet->state,
        chain_data->error_index, chain_data->submit_index);
  }
  return iree_ok_status();
}
