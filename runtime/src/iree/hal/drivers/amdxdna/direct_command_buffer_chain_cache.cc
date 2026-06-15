// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "iree/hal/drivers/amdxdna/device_internal.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

static std::mutex& iree_hal_amdxdna_chain_command_cache_init_mutex() {
  static std::mutex mutex;
  return mutex;
}

iree_hal_amdxdna_device_chain_command_cache_t*
iree_hal_amdxdna_get_chain_command_cache(iree_hal_amdxdna_device* device) {
  if (device->chain_command_cache) {
    return device->chain_command_cache;
  }
  std::lock_guard<std::mutex> lock(
      iree_hal_amdxdna_chain_command_cache_init_mutex());
  if (!device->chain_command_cache) {
    device->chain_command_cache =
        new iree_hal_amdxdna_device_chain_command_cache_t();
  }
  return device->chain_command_cache;
}

void iree_hal_amdxdna_device_destroy_chain_command_cache(
    iree_hal_amdxdna_device* device) {
  delete device->chain_command_cache;
  device->chain_command_cache = nullptr;
}

bool iree_hal_amdxdna_direct_command_buffer_control_words_changed(
    const uint32_t* cached_words, iree_host_size_t cached_word_count,
    const uint32_t* fresh_words, iree_host_size_t fresh_word_count) {
  if (cached_word_count != fresh_word_count) return true;
  if (cached_word_count == 0) return false;
  return std::memcmp(cached_words, fresh_words,
                     cached_word_count * sizeof(uint32_t)) != 0;
}

static bool iree_hal_amdxdna_chain_cmd_device_signature_matches(
    const iree_hal_amdxdna_chain_cmd& lhs,
    const iree_hal_amdxdna_chain_cmd& rhs) {
  return lhs.ctrl_words == rhs.ctrl_words &&
         lhs.binding_buffers == rhs.binding_buffers &&
         lhs.binding_device_addrs == rhs.binding_device_addrs &&
         lhs.binding_offsets == rhs.binding_offsets &&
         lhs.binding_lengths == rhs.binding_lengths;
}

static bool iree_hal_amdxdna_chain_cmd_shape_matches(
    const iree_hal_amdxdna_chain_cmd& lhs,
    const iree_hal_amdxdna_chain_cmd& rhs) {
  return lhs.ctrl_words.size() == rhs.ctrl_words.size() &&
         lhs.binding_buffers.size() == rhs.binding_buffers.size() &&
         lhs.binding_offsets == rhs.binding_offsets &&
         lhs.binding_lengths == rhs.binding_lengths;
}

size_t iree_hal_amdxdna_chain_group_logical_command_count(
    const iree_hal_amdxdna_chain_group& group) {
  size_t total = 0;
  for (const iree_hal_amdxdna_chain_cmd& cmd : group.cmds) {
    const size_t repeat_count = std::max<size_t>(cmd.repeat_count, 1);
    if (repeat_count > std::numeric_limits<size_t>::max() - total) {
      return std::numeric_limits<size_t>::max();
    }
    total += repeat_count;
  }
  return total;
}

static const iree_hal_amdxdna_chain_cmd*
iree_hal_amdxdna_chain_group_cmd_at_logical_index(
    const iree_hal_amdxdna_chain_group& group, size_t logical_index) {
  size_t cursor = 0;
  for (const iree_hal_amdxdna_chain_cmd& cmd : group.cmds) {
    const size_t repeat_count = std::max<size_t>(cmd.repeat_count, 1);
    if (logical_index < cursor + repeat_count) return &cmd;
    cursor += repeat_count;
  }
  return nullptr;
}

bool iree_hal_amdxdna_chain_command_cache_device_matches(
    const iree_hal_amdxdna_chain_command_cache_entry& cache,
    const iree_hal_amdxdna_chain_group& group, uint32_t max_slots) {
  if (cache.chains.empty() || cache.max_slots != max_slots ||
      cache.group.queue != group.queue ||
      cache.group.native_partial_elf != group.native_partial_elf ||
      cache.group.cmds.size() != group.cmds.size() ||
      !cache.group.reconf_buffers.empty() || !group.reconf_buffers.empty()) {
    return false;
  }
  for (size_t i = 0; i < group.cmds.size(); ++i) {
    if (!iree_hal_amdxdna_chain_cmd_device_signature_matches(
            cache.group.cmds[i], group.cmds[i])) {
      return false;
    }
  }
  return true;
}

bool iree_hal_amdxdna_chain_command_cache_shape_matches(
    const iree_hal_amdxdna_chain_command_cache_entry& cache,
    const iree_hal_amdxdna_chain_group& group, uint32_t max_slots) {
  if (cache.group.cmds.empty() || cache.max_slots != max_slots ||
      cache.group.queue != group.queue ||
      cache.group.native_partial_elf != group.native_partial_elf ||
      cache.group.cmds.size() != group.cmds.size() ||
      !cache.group.reconf_buffers.empty() || !group.reconf_buffers.empty()) {
    return false;
  }
  for (size_t i = 0; i < group.cmds.size(); ++i) {
    if (!iree_hal_amdxdna_chain_cmd_shape_matches(cache.group.cmds[i],
                                                  group.cmds[i])) {
      return false;
    }
  }
  return true;
}

// Deferred-build exact match: compares the inputs that deterministically
// produce a built child (control-code template pointer + constants + cu +
// bindings) so flush can reuse an already-built cached chain WITHOUT building
// this group's deferred children. Equivalent to the ctrl_words signature match
// but computable on unbuilt descriptors.
bool iree_hal_amdxdna_chain_cmd_descriptor_matches(
    const iree_hal_amdxdna_chain_cmd& lhs,
    const iree_hal_amdxdna_chain_cmd& rhs) {
  return lhs.src_asm_inst == rhs.src_asm_inst &&
         lhs.src_patches == rhs.src_patches &&
         lhs.src_use_native_partial_elf == rhs.src_use_native_partial_elf &&
         lhs.src_cu_idx.index == rhs.src_cu_idx.index &&
         lhs.src_constants == rhs.src_constants &&
         lhs.binding_buffers == rhs.binding_buffers &&
         lhs.binding_device_addrs == rhs.binding_device_addrs &&
         lhs.binding_offsets == rhs.binding_offsets &&
         lhs.binding_lengths == rhs.binding_lengths;
}

bool iree_hal_amdxdna_chain_command_cache_descriptor_matches(
    const iree_hal_amdxdna_chain_command_cache_entry& cache,
    const iree_hal_amdxdna_chain_group& group, uint32_t max_slots) {
  const size_t cache_logical_count =
      iree_hal_amdxdna_chain_group_logical_command_count(cache.group);
  const size_t group_logical_count =
      iree_hal_amdxdna_chain_group_logical_command_count(group);
  if (cache.chains.empty() || cache.max_slots != max_slots ||
      cache.group.queue != group.queue ||
      cache.group.native_partial_elf != group.native_partial_elf ||
      cache_logical_count != group_logical_count ||
      !cache.group.reconf_buffers.empty() || !group.reconf_buffers.empty()) {
    return false;
  }
  for (size_t i = 0; i < group_logical_count; ++i) {
    const iree_hal_amdxdna_chain_cmd* cache_cmd =
        iree_hal_amdxdna_chain_group_cmd_at_logical_index(cache.group, i);
    const iree_hal_amdxdna_chain_cmd* group_cmd =
        iree_hal_amdxdna_chain_group_cmd_at_logical_index(group, i);
    if (!cache_cmd || !group_cmd) return false;
    if (!iree_hal_amdxdna_chain_cmd_descriptor_matches(*cache_cmd,
                                                       *group_cmd)) {
      return false;
    }
  }
  return true;
}

iree_status_t iree_hal_amdxdna_update_cached_chain_cmd(
    iree_hal_amdxdna_chain_cmd& cached, const iree_hal_amdxdna_chain_cmd& fresh,
    bool* out_packet_changed, bool* out_code_changed,
    bool* out_device_bindings_changed, bool* out_rebound) {
  if (out_packet_changed) *out_packet_changed = false;
  if (out_code_changed) *out_code_changed = false;
  if (out_device_bindings_changed) *out_device_bindings_changed = false;
  if (out_rebound) *out_rebound = false;
  if (!iree_hal_amdxdna_chain_cmd_shape_matches(cached, fresh)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna cached native chain command shape changed");
  }
  const bool code_changed =
      iree_hal_amdxdna_direct_command_buffer_control_words_changed(
          cached.ctrl_words.data(), cached.ctrl_words.size(),
          fresh.ctrl_words.data(), fresh.ctrl_words.size());
  const bool device_bindings_changed =
      cached.binding_device_addrs != fresh.binding_device_addrs ||
      cached.binding_offsets != fresh.binding_offsets ||
      cached.binding_lengths != fresh.binding_lengths;
  const bool native_bindings_changed =
      cached.binding_buffers != fresh.binding_buffers ||
      device_bindings_changed || !cached.native_bindings_current;
  if (code_changed) {
    void* ctrl_ptr = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_buffer_map(cached.ctrl_code.get(), &ctrl_ptr));
    std::memcpy(ctrl_ptr, fresh.ctrl_words.data(),
                fresh.ctrl_words.size() * sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_mark_code_dirty(cached.command.get()));
    cached.ctrl_words = fresh.ctrl_words;
  }
  if (native_bindings_changed) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_reset_bound_buffers(
        cached.command.get()));
    for (size_t i = 0; i < fresh.binding_buffers.size(); ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
          cached.command.get(), /*position=*/i + 1, fresh.binding_buffers[i],
          fresh.binding_offsets[i], fresh.binding_lengths[i]));
    }
    cached.native_bindings_current = true;
    if (out_rebound) *out_rebound = true;
  }
  cached.binding_buffers = fresh.binding_buffers;
  cached.binding_device_addrs = fresh.binding_device_addrs;
  cached.binding_offsets = fresh.binding_offsets;
  cached.binding_lengths = fresh.binding_lengths;
  if (out_packet_changed) {
    *out_packet_changed = code_changed || device_bindings_changed ||
                          native_bindings_changed;
  }
  if (out_code_changed) *out_code_changed = code_changed;
  if (out_device_bindings_changed) {
    *out_device_bindings_changed = device_bindings_changed;
  }
  return iree_ok_status();
}
