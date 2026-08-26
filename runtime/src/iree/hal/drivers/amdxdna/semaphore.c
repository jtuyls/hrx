// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host-side timeline semaphore aligned with
// iree/hal/drivers/local_sync/sync_semaphore.c (no GPU primitive).

#include "iree/hal/drivers/amdxdna/semaphore.h"

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/completion_queue.h"
#include "iree/hal/semaphore.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdxdna_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdxdna_semaphore_t {
  iree_async_semaphore_t async;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_completion_batch_t* native_signal_batch;
  uint64_t native_signal_value;
  iree_hal_semaphore_t* native_issue_semaphore;
  iree_allocator_t host_allocator;
} iree_hal_amdxdna_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_amdxdna_semaphore_vtable;

static iree_hal_amdxdna_completion_batch_t*
iree_hal_amdxdna_semaphore_try_retain_native_signal(
    iree_hal_amdxdna_semaphore_t* semaphore, uint64_t value) {
  iree_hal_amdxdna_completion_batch_t* native_signal_batch = NULL;
  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->native_signal_batch &&
      semaphore->native_signal_value == value) {
    native_signal_batch = iree_hal_amdxdna_completion_batch_retain(
        semaphore->native_signal_batch);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  return native_signal_batch;
}

static iree_hal_amdxdna_semaphore_t* iree_hal_amdxdna_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdxdna_semaphore_vtable);
  return (iree_hal_amdxdna_semaphore_t*)base_value;
}

bool iree_hal_amdxdna_semaphore_list_is_ready_or_native_orderable(
    iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_t* base_semaphore = semaphore_list.semaphores[i];
    if (!iree_hal_amdxdna_semaphore_isa(base_semaphore)) return false;
    iree_hal_amdxdna_semaphore_t* semaphore =
        iree_hal_amdxdna_semaphore_cast(base_semaphore);
    const uint64_t value = semaphore_list.payload_values[i];
    const uint64_t current = iree_async_semaphore_query(&semaphore->async);
    if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) return false;
    if (current >= value) continue;

    iree_hal_amdxdna_completion_batch_t* native_signal_batch =
        iree_hal_amdxdna_semaphore_try_retain_native_signal(semaphore, value);
    const bool native_orderable =
        iree_hal_amdxdna_completion_batch_is_native_orderable(
            native_signal_batch);
    iree_hal_amdxdna_completion_batch_destroy(native_signal_batch);
    if (!native_orderable) return false;
  }
  return true;
}

iree_status_t iree_hal_amdxdna_semaphore_list_clone_native_issue_waits(
    iree_hal_semaphore_list_t semaphore_list, iree_allocator_t host_allocator,
    bool* out_available, iree_hal_semaphore_list_t* out_list) {
  IREE_ASSERT_ARGUMENT(out_available);
  IREE_ASSERT_ARGUMENT(out_list);
  *out_available = false;
  *out_list = iree_hal_semaphore_list_empty();
  if (semaphore_list.count == 0) {
    *out_available = true;
    return iree_ok_status();
  }

  iree_hal_semaphore_t** issue_semaphores = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, semaphore_list.count, sizeof(*issue_semaphores),
      (void**)&issue_semaphores));
  bool available = true;
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_t* base_semaphore = semaphore_list.semaphores[i];
    if (!iree_hal_amdxdna_semaphore_isa(base_semaphore)) {
      available = false;
      break;
    }
    iree_hal_amdxdna_semaphore_t* semaphore =
        iree_hal_amdxdna_semaphore_cast(base_semaphore);
    const uint64_t value = semaphore_list.payload_values[i];
    const uint64_t current = iree_async_semaphore_query(&semaphore->async);
    if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      available = false;
      break;
    }
    if (current >= value) {
      issue_semaphores[i] = base_semaphore;
      continue;
    }
    iree_hal_amdxdna_completion_batch_t* native_signal_batch =
        iree_hal_amdxdna_semaphore_try_retain_native_signal(semaphore, value);
    available = native_signal_batch && semaphore->native_issue_semaphore &&
                iree_hal_amdxdna_completion_batch_has_native_issue(
                    native_signal_batch);
    iree_hal_amdxdna_completion_batch_destroy(native_signal_batch);
    if (!available) break;
    issue_semaphores[i] = semaphore->native_issue_semaphore;
  }

  iree_status_t status = iree_ok_status();
  if (available) {
    iree_hal_semaphore_list_t issue_list = {
        .count = semaphore_list.count,
        .semaphores = issue_semaphores,
        .payload_values = semaphore_list.payload_values,
    };
    status = iree_hal_semaphore_list_clone(&issue_list, host_allocator,
                                           out_list);
    if (iree_status_is_ok(status)) *out_available = true;
  }
  iree_allocator_free(host_allocator, issue_semaphores);
  return status;
}

void iree_hal_amdxdna_semaphore_list_signal_native_issue(
    iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!iree_hal_amdxdna_semaphore_isa(semaphore_list.semaphores[i])) continue;
    iree_hal_amdxdna_semaphore_t* semaphore =
        iree_hal_amdxdna_semaphore_cast(semaphore_list.semaphores[i]);
    if (!semaphore->native_issue_semaphore) continue;
    iree_status_ignore(iree_hal_semaphore_signal(
        semaphore->native_issue_semaphore, semaphore_list.payload_values[i],
        /*frontier=*/NULL));
  }
}

void iree_hal_amdxdna_semaphore_list_fail_native_issue(
    iree_hal_semaphore_list_t semaphore_list, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!iree_hal_amdxdna_semaphore_isa(semaphore_list.semaphores[i])) continue;
    iree_hal_amdxdna_semaphore_t* semaphore =
        iree_hal_amdxdna_semaphore_cast(semaphore_list.semaphores[i]);
    if (!semaphore->native_issue_semaphore) continue;
    iree_hal_semaphore_fail(semaphore->native_issue_semaphore,
                            iree_status_clone(status));
  }
  iree_status_ignore(status);
}

iree_status_t iree_hal_amdxdna_semaphore_list_retain_native_wait_batches(
    iree_hal_semaphore_list_t semaphore_list, iree_allocator_t host_allocator,
    iree_host_size_t* out_count,
    iree_hal_amdxdna_completion_batch_t*** out_batches) {
  IREE_ASSERT_ARGUMENT(out_count);
  IREE_ASSERT_ARGUMENT(out_batches);
  *out_count = 0;
  *out_batches = NULL;
  if (semaphore_list.count == 0) return iree_ok_status();

  iree_hal_amdxdna_completion_batch_t** batches = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, semaphore_list.count, sizeof(*batches),
      (void**)&batches));
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_t* base_semaphore = semaphore_list.semaphores[i];
    if (!iree_hal_amdxdna_semaphore_isa(base_semaphore)) continue;
    iree_hal_amdxdna_semaphore_t* semaphore =
        iree_hal_amdxdna_semaphore_cast(base_semaphore);
    const uint64_t value = semaphore_list.payload_values[i];
    if (iree_async_semaphore_query(&semaphore->async) >= value) continue;
    iree_hal_amdxdna_completion_batch_t* batch =
        iree_hal_amdxdna_semaphore_try_retain_native_signal(semaphore, value);
    if (batch) batches[(*out_count)++] = batch;
  }
  if (*out_count == 0) {
    iree_allocator_free(host_allocator, batches);
  } else {
    *out_batches = batches;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_semaphore_allocate(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, bool create_native_issue_semaphore,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;
  (void)flags;

  iree_hal_amdxdna_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(sizeof(*semaphore), 0, &frontier_offset,
                                      &total_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_amdxdna_semaphore_vtable,
      proactor, initial_value, frontier_offset, 0, &semaphore->async);
  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->native_signal_batch = NULL;
  semaphore->native_signal_value = 0;
  semaphore->native_issue_semaphore = NULL;
  semaphore->host_allocator = host_allocator;
  if (create_native_issue_semaphore) {
    iree_status_t status = iree_hal_amdxdna_semaphore_allocate(
        proactor, queue_affinity, initial_value, flags, host_allocator,
        /*create_native_issue_semaphore=*/false,
        &semaphore->native_issue_semaphore);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_deinitialize(&semaphore->mutex);
      iree_async_semaphore_deinitialize(&semaphore->async);
      iree_allocator_free(host_allocator, semaphore);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }
  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_amdxdna_semaphore_allocate(
      proactor, queue_affinity, initial_value, flags, host_allocator,
      /*create_native_issue_semaphore=*/true, out_semaphore);
}

static void iree_hal_amdxdna_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_completion_batch_t* native_signal_batch = NULL;
  iree_slim_mutex_lock(&semaphore->mutex);
  native_signal_batch = semaphore->native_signal_batch;
  semaphore->native_signal_batch = NULL;
  semaphore->native_signal_value = 0;
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_amdxdna_completion_batch_destroy(native_signal_batch);
  iree_hal_semaphore_release(semaphore->native_issue_semaphore);
  iree_slim_mutex_deinitialize(&semaphore->mutex);
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdxdna_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amdxdna_semaphore_vtable);
}

void iree_hal_amdxdna_semaphore_record_native_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (!iree_hal_amdxdna_semaphore_isa(base_semaphore) || !batch) return;
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(base_semaphore);
  iree_hal_amdxdna_completion_batch_t* old_batch = NULL;
  iree_hal_amdxdna_completion_batch_retain(batch);
  iree_slim_mutex_lock(&semaphore->mutex);
  old_batch = semaphore->native_signal_batch;
  semaphore->native_signal_batch = batch;
  semaphore->native_signal_value = value;
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_amdxdna_completion_batch_destroy(old_batch);
}

void iree_hal_amdxdna_semaphore_clear_native_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (!iree_hal_amdxdna_semaphore_isa(base_semaphore) || !batch) return;
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(base_semaphore);
  iree_hal_amdxdna_completion_batch_t* old_batch = NULL;
  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->native_signal_batch == batch &&
      semaphore->native_signal_value == value) {
    old_batch = semaphore->native_signal_batch;
    semaphore->native_signal_batch = NULL;
    semaphore->native_signal_value = 0;
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_amdxdna_completion_batch_destroy(old_batch);
}

static uint64_t iree_hal_amdxdna_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  // Same as iree_hal_sync_semaphore_query: failure_status is stored as intptr.
  iree_status_t failure = (iree_status_t)(uintptr_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_amdxdna_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) return status;
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(base_semaphore);
  uint64_t current = iree_async_semaphore_query(&semaphore->async);
  if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    return iree_hal_semaphore_failure_as_status(current);
  }
  if (current >= value) {
    return iree_ok_status();
  }

  iree_hal_amdxdna_completion_batch_t* native_signal_batch = NULL;
  if (iree_timeout_is_infinite(timeout)) {
    native_signal_batch =
        iree_hal_amdxdna_semaphore_try_retain_native_signal(semaphore, value);
  }
  if (native_signal_batch) {
    iree_status_t status = iree_hal_amdxdna_completion_batch_wait(
        native_signal_batch, timeout, flags);
    iree_hal_amdxdna_completion_batch_destroy(native_signal_batch);
    if (!iree_status_is_ok(status)) {
      return status;
    }

    current = iree_async_semaphore_query(&semaphore->async);
    if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      return iree_hal_semaphore_failure_as_status(current);
    }
    if (current >= value) {
      return iree_ok_status();
    }
  }

  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
}

static iree_status_t iree_hal_amdxdna_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)external_timepoint;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_amdxdna_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)requested_type;
  (void)requested_flags;
  (void)out_external_timepoint;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_amdxdna_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_amdxdna_semaphore_destroy,
            .query = iree_hal_amdxdna_semaphore_query,
            .signal = iree_hal_amdxdna_semaphore_signal,
            .on_fail = NULL,
        },
    .wait = iree_hal_amdxdna_semaphore_wait,
    .import_timepoint = iree_hal_amdxdna_semaphore_import_timepoint,
    .export_timepoint = iree_hal_amdxdna_semaphore_export_timepoint,
};
