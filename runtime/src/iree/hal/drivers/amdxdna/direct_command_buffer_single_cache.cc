// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"

#include <algorithm>
#include <cstring>

#include "iree/hal/drivers/amdxdna/device_internal.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

std::mutex& iree_hal_amdxdna_single_command_cache_init_mutex() {
  static std::mutex mutex;
  return mutex;
}

iree_hal_amdxdna_device_single_command_cache_t*
iree_hal_amdxdna_get_single_command_cache(iree_hal_amdxdna_device* device) {
  if (device->single_command_cache) {
    return device->single_command_cache;
  }
  std::lock_guard<std::mutex> lock(
      iree_hal_amdxdna_single_command_cache_init_mutex());
  if (!device->single_command_cache) {
    device->single_command_cache =
        new iree_hal_amdxdna_device_single_command_cache_t();
  }
  return device->single_command_cache;
}

void iree_hal_amdxdna_device_destroy_single_command_cache(
    iree_hal_amdxdna_device* device) {
  delete device->single_command_cache;
  device->single_command_cache = nullptr;
}

bool iree_hal_amdxdna_single_command_cache_matches(
    const iree_hal_amdxdna_single_command_cache_entry& cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const std::vector<uint32_t>& ctrl_words,
    const std::vector<iree_hal_amdxdna_native_buffer_t*>& binding_buffers,
    const std::vector<uint64_t>& binding_device_addrs,
    const std::vector<iree_device_size_t>& binding_offsets,
    const std::vector<iree_device_size_t>& binding_lengths) {
  return cache.command && cache.queue == queue && cache.cu_index == cu_index &&
         cache.ctrl_words == ctrl_words &&
         cache.binding_buffers == binding_buffers &&
         cache.binding_device_addrs == binding_device_addrs &&
         cache.binding_offsets == binding_offsets &&
         cache.binding_lengths == binding_lengths;
}

bool iree_hal_amdxdna_single_command_cache_shape_matches(
    const iree_hal_amdxdna_single_command_cache_entry& cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const std::vector<uint32_t>& ctrl_words,
    const std::vector<iree_device_size_t>& binding_offsets,
    const std::vector<iree_device_size_t>& binding_lengths) {
  return cache.command && cache.queue == queue && cache.cu_index == cu_index &&
         cache.ctrl_words.size() == ctrl_words.size() &&
         cache.binding_offsets == binding_offsets &&
         cache.binding_lengths == binding_lengths;
}

iree_status_t iree_hal_amdxdna_update_single_command_cache_entry(
    iree_hal_amdxdna_single_command_cache_entry& cache,
    const std::vector<uint32_t>& ctrl_words,
    const std::vector<iree_hal_amdxdna_native_buffer_t*>& binding_buffers,
    const std::vector<uint64_t>& binding_device_addrs,
    const std::vector<iree_device_size_t>& binding_offsets,
    const std::vector<iree_device_size_t>& binding_lengths) {
  const bool ctrl_changed = cache.ctrl_words != ctrl_words;
  const bool bindings_changed = cache.binding_buffers != binding_buffers ||
                                cache.binding_offsets != binding_offsets ||
                                cache.binding_lengths != binding_lengths;
  if (ctrl_changed) {
    void* ctrl_ptr = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(
        cache.ctrl_code_buffer.get(), &ctrl_ptr));
    std::memcpy(ctrl_ptr, ctrl_words.data(),
                ctrl_words.size() * sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
        cache.ctrl_code_buffer.get(),
        iree_hal_amdxdna_native_sync_direction_t::host_to_device));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_mark_code_dirty(cache.command.get()));
    cache.ctrl_words = ctrl_words;
  }
  if (bindings_changed) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_reset_bound_buffers(
        cache.command.get()));
    for (size_t i = 0; i < binding_buffers.size(); ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
          cache.command.get(), /*position=*/i + 1, binding_buffers[i],
          binding_offsets[i], binding_lengths[i]));
    }
    cache.binding_buffers = binding_buffers;
  }
  cache.binding_device_addrs = binding_device_addrs;
  cache.binding_offsets = binding_offsets;
  cache.binding_lengths = binding_lengths;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_find_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const std::vector<uint32_t>& ctrl_words,
    const std::vector<iree_hal_amdxdna_native_buffer_t*>& binding_buffers,
    const std::vector<uint64_t>& binding_device_addrs,
    const std::vector<iree_device_size_t>& binding_offsets,
    const std::vector<iree_device_size_t>& binding_lengths,
    iree_hal_amdxdna_single_command_cache_entry** out_entry) {
  *out_entry = nullptr;
  for (auto& entry : cache->entries) {
    if (iree_hal_amdxdna_single_command_cache_matches(
            entry, queue, cu_index, ctrl_words, binding_buffers,
            binding_device_addrs, binding_offsets, binding_lengths)) {
      entry.last_use = ++cache->use_clock;
      *out_entry = &entry;
      return iree_ok_status();
    }
  }
  for (auto& entry : cache->entries) {
    if (iree_hal_amdxdna_single_command_cache_shape_matches(
            entry, queue, cu_index, ctrl_words, binding_offsets,
            binding_lengths)) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_update_single_command_cache_entry(
          entry, ctrl_words, binding_buffers, binding_device_addrs,
          binding_offsets, binding_lengths));
      entry.last_use = ++cache->use_clock;
      *out_entry = &entry;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_hal_amdxdna_single_command_cache_entry*
iree_hal_amdxdna_store_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    std::vector<uint32_t> ctrl_words,
    std::vector<iree_hal_amdxdna_native_buffer_t*> binding_buffers,
    std::vector<uint64_t> binding_device_addrs,
    std::vector<iree_device_size_t> binding_offsets,
    std::vector<iree_device_size_t> binding_lengths,
    iree_hal_amdxdna_native_buffer_ptr ctrl_code_buffer,
    iree_hal_amdxdna_native_command_ptr command) {
  if (cache->entries.size() >= kAmdxdnaSingleCommandCacheCapacity) {
    auto lru = std::min_element(
        cache->entries.begin(), cache->entries.end(),
        [](const iree_hal_amdxdna_single_command_cache_entry& lhs,
           const iree_hal_amdxdna_single_command_cache_entry& rhs) {
          return lhs.last_use < rhs.last_use;
        });
    *lru = iree_hal_amdxdna_single_command_cache_entry();
    cache->entries.erase(lru);
  }
  cache->entries.emplace_back();
  auto& entry = cache->entries.back();
  entry.queue = queue;
  entry.cu_index = cu_index;
  entry.ctrl_words = std::move(ctrl_words);
  entry.binding_buffers = std::move(binding_buffers);
  entry.binding_device_addrs = std::move(binding_device_addrs);
  entry.binding_offsets = std::move(binding_offsets);
  entry.binding_lengths = std::move(binding_lengths);
  entry.ctrl_code_buffer = std::move(ctrl_code_buffer);
  entry.command = std::move(command);
  entry.last_use = ++cache->use_clock;
  return &entry;
}
