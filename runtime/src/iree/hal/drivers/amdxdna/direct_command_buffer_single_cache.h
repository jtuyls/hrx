// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

// Device-global cache of prepared single-dispatch native commands (ERT_START_CU
// / START_NPU path), reused across one-shot command-buffer instances and
// rebound/rewritten in place to match a freshly recorded dispatch.
//
// Ownership/threading contract: like the chain command cache, this is
// intentionally DEVICE-GLOBAL. On the PARTIAL_ELF dispatch path (the only
// single-dispatch path the Windows MCDM backend enables) the cache mutex is
// held across the cached command's submit, so it ALSO serializes the
// aperture-touching submit on a context (required by the native.h Windows MCDM
// contract). The non-PARTIAL_ELF single path (Linux KMQ) does not stage into a
// shared aperture and is intentionally unlocked. Re-owning prepared commands
// per executable entry point would need a separate per-context submit lock to
// preserve that serialization on the MCDM path; that is tracked as the
// prepared-command revisit in the native-DDI follow-ups doc.

struct iree_hal_amdxdna_device;

struct iree_hal_amdxdna_single_command_cache_entry {
  iree_hal_amdxdna_native_queue_t* queue = nullptr;
  uint32_t cu_index = 0;
  std::vector<uint32_t> ctrl_words;
  std::vector<iree_hal_amdxdna_native_buffer_t*> binding_buffers;
  std::vector<uint64_t> binding_device_addrs;
  std::vector<iree_device_size_t> binding_offsets;
  std::vector<iree_device_size_t> binding_lengths;
  iree_hal_amdxdna_native_buffer_ptr ctrl_code_buffer;
  iree_hal_amdxdna_native_command_ptr command;
  uint64_t last_use = 0;
};

constexpr size_t kAmdxdnaSingleCommandCacheCapacity = 8;

struct iree_hal_amdxdna_device_single_command_cache_t {
  std::mutex mutex;
  std::vector<iree_hal_amdxdna_single_command_cache_entry> entries;
  uint64_t use_clock = 0;
};

iree_hal_amdxdna_device_single_command_cache_t*
iree_hal_amdxdna_get_single_command_cache(iree_hal_amdxdna_device* device);

iree_status_t iree_hal_amdxdna_find_single_command_cache_entry(
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_native_queue_t* queue, uint32_t cu_index,
    const std::vector<uint32_t>& ctrl_words,
    const std::vector<iree_hal_amdxdna_native_buffer_t*>& binding_buffers,
    const std::vector<uint64_t>& binding_device_addrs,
    const std::vector<iree_device_size_t>& binding_offsets,
    const std::vector<iree_device_size_t>& binding_lengths,
    iree_hal_amdxdna_single_command_cache_entry** out_entry);

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
    iree_hal_amdxdna_native_command_ptr command);

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_SINGLE_CACHE_H_
