// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_CHAIN_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_CHAIN_CACHE_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

// Chain (ERT_CMD_CHAIN / runlist) building blocks and the device-global chain
// command cache.
//
// Ownership/threading contract: the chain command cache is intentionally
// DEVICE-GLOBAL, not executable- or command-buffer-owned. A chain spans
// multiple executable entry points (a sequence of dispatches), so no single
// entry point can own it; and its whole value is reuse across one-shot
// command-buffer instances of the same shape, which only a device-scoped cache
// provides. Entries are mutable (control words rewritten / buffers rebound to
// match a freshly recorded command) and are serialized by the cache mutex; a
// cached chain may be reused by one submission lane at a time. See the
// native-DDI follow-ups doc ("Prepared-command model: scope decision") for why
// this stays device-global and what would let it change.

struct iree_hal_amdxdna_device;

// One chainable command: a host-patched control-code BO + its ERT_START_NPU
// native command. Kept alive (buffer + command) until the chain completes.
struct iree_hal_amdxdna_chain_cmd {
  iree_hal_amdxdna_native_buffer_ptr ctrl_code;
  iree_hal_amdxdna_native_command_ptr command;
  std::vector<uint32_t> ctrl_words;
  std::vector<iree_hal_amdxdna_native_buffer_t*> binding_buffers;
  std::vector<uint64_t> binding_device_addrs;
  std::vector<iree_device_size_t> binding_offsets;
  std::vector<iree_device_size_t> binding_lengths;
  // Deferred-build inputs (set by accumulate). The ctrl_code BO + native
  // command above are built lazily on a cache miss in flush, so a steady-state
  // exact cache hit reuses the already-built cached chain and skips the
  // per-child build entirely (XRT's build-once-reuse model). `src_asm_inst` is
  // a stable pointer into the executable's immutable control-code words;
  // together with the constants and binding addresses it determines the patched
  // `ctrl_words`, so the exact-match fast path compares these cheap inputs
  // instead of building and comparing the streams.
  const std::vector<uint32_t>* src_asm_inst = nullptr;
  const std::vector<uint32_t>* src_patches = nullptr;
  std::vector<uint8_t> src_constants;
  iree_hal_amdxdna_native_cu_index_t src_cu_idx{};
  bool src_use_native_partial_elf = false;
  bool built = false;
  // Number of consecutive logical child commands represented by this
  // descriptor during deferred replay. Cached native chains are stored expanded
  // (one native child command per logical run), but an incoming replay can stay
  // compact until it misses the cache.
  size_t repeat_count = 1;
  // False when metadata was refreshed for a device-visible hit but the native
  // child command still holds older bound-buffer pointers. Safe until the next
  // packet rewrite, which must rebind before BO-table generation dereferences.
  bool native_bindings_current = true;
};

// A contiguous run of dispatches that share one native queue. Flushed as one
// ERT_CMD_CHAIN (split into multiple chains only if the slot count exceeds the
// exec buffer). A chain runs on a single native context, so a queue change
// between dispatches starts a new group.
struct iree_hal_amdxdna_chain_group {
  // Retains the native context owning `queue` until the chain flushes.
  std::shared_ptr<iree_hal_amdxdna_native_context_t> context;
  iree_hal_amdxdna_native_queue_t* queue = nullptr;
  std::vector<iree_hal_amdxdna_chain_cmd> cmds;
  // Control-packet sequence BOs (reconfig arg buffers): referenced by address
  // from the slots, kept alive + bound for residency until the chain completes.
  std::vector<iree_hal_amdxdna_native_buffer_ptr> reconf_buffers;
  // I/O binding refs: their BOs are bound for residency and their exact ranges
  // are synced device->host after the chain completes.
  std::vector<iree_hal_buffer_ref_t> binding_refs;
  // True when the native context already loaded the PDI and dispatches should
  // use module-style START_NPU/PARTIAL_ELF child packets. These children carry
  // their own BO tables, so the parent chain binds child command BOs instead
  // of binding every dereferenced BO itself.
  bool native_partial_elf = false;
};

// Accumulates sub-commands across dispatches so a whole command buffer flushes
// per native queue. Native backends that support command chaining flush this as
// ERT_CMD_CHAIN.
struct iree_hal_amdxdna_chain_accum {
  std::vector<iree_hal_amdxdna_chain_group> groups;
};

struct iree_hal_amdxdna_chain_command_cache_entry {
  iree_hal_amdxdna_chain_group group;
  uint32_t max_slots = 0;
  std::vector<iree_hal_amdxdna_native_command_ptr> chains;
  uint64_t last_use = 0;
};

static constexpr size_t kAmdxdnaChainCommandCacheCapacity = 4;

struct iree_hal_amdxdna_device_chain_command_cache_t {
  std::mutex mutex;
  std::vector<iree_hal_amdxdna_chain_command_cache_entry> entries;
  uint64_t use_clock = 0;
};

iree_hal_amdxdna_device_chain_command_cache_t*
iree_hal_amdxdna_get_chain_command_cache(iree_hal_amdxdna_device* device);

size_t iree_hal_amdxdna_chain_group_logical_command_count(
    const iree_hal_amdxdna_chain_group& group);

bool iree_hal_amdxdna_chain_cmd_descriptor_matches(
    const iree_hal_amdxdna_chain_cmd& lhs,
    const iree_hal_amdxdna_chain_cmd& rhs);

bool iree_hal_amdxdna_chain_command_cache_device_matches(
    const iree_hal_amdxdna_chain_command_cache_entry& cache,
    const iree_hal_amdxdna_chain_group& group, uint32_t max_slots);

bool iree_hal_amdxdna_chain_command_cache_shape_matches(
    const iree_hal_amdxdna_chain_command_cache_entry& cache,
    const iree_hal_amdxdna_chain_group& group, uint32_t max_slots);

bool iree_hal_amdxdna_chain_command_cache_descriptor_matches(
    const iree_hal_amdxdna_chain_command_cache_entry& cache,
    const iree_hal_amdxdna_chain_group& group, uint32_t max_slots);

iree_status_t iree_hal_amdxdna_update_cached_chain_cmd(
    iree_hal_amdxdna_chain_cmd& cached, const iree_hal_amdxdna_chain_cmd& fresh,
    bool* out_packet_changed, bool* out_code_changed,
    bool* out_device_bindings_changed, bool* out_rebound);

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_CHAIN_CACHE_H_
