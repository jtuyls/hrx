// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

struct iree_async_proactor_t;
struct iree_hal_amdxdna_completion_batch_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Software timeline semaphore (same behavior as IREE local_sync). Required for
// correct host-side wait/signal with the async HAL.
iree_status_t iree_hal_amdxdna_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

bool iree_hal_amdxdna_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns true when every wait is either already satisfied or backed by an
// already-issued native completion batch. This is only an issue-ordering
// predicate; it does not advance any software semaphore timeline.
bool iree_hal_amdxdna_semaphore_list_is_ready_or_native_orderable(
    iree_hal_semaphore_list_t semaphore_list);

// Clones |semaphore_list| with unresolved native-produced waits replaced by
// their internal issue timelines. |out_available| is false when any unresolved
// wait has no promised native producer; |out_list| is empty in that case.
iree_status_t iree_hal_amdxdna_semaphore_list_clone_native_issue_waits(
    iree_hal_semaphore_list_t semaphore_list, iree_allocator_t host_allocator,
    bool* out_available, iree_hal_semaphore_list_t* out_list);

void iree_hal_amdxdna_semaphore_list_signal_native_issue(
    iree_hal_semaphore_list_t semaphore_list);
void iree_hal_amdxdna_semaphore_list_fail_native_issue(
    iree_hal_semaphore_list_t semaphore_list, iree_status_t status);

// Retains native producer batches for unresolved waits. The caller owns the
// returned array and each retained batch.
iree_status_t iree_hal_amdxdna_semaphore_list_retain_native_wait_batches(
    iree_hal_semaphore_list_t semaphore_list, iree_allocator_t host_allocator,
    iree_host_size_t* out_count,
    struct iree_hal_amdxdna_completion_batch_t*** out_batches);

// Publishes/clears the exact native completion batch expected to signal
// |semaphore| to |value|. Explicit host waits may use this to wait the native
// completion path directly instead of waiting for the software semaphore path.
void iree_hal_amdxdna_semaphore_record_native_signal(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    struct iree_hal_amdxdna_completion_batch_t* batch);
void iree_hal_amdxdna_semaphore_clear_native_signal(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    struct iree_hal_amdxdna_completion_batch_t* batch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_SEMAPHORE_H_
