// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/completion_queue.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/processor.h"
#include "iree/base/threading/thread.h"
#include "iree/base/time.h"
#include "iree/hal/drivers/amdxdna/semaphore.h"

typedef enum iree_hal_amdxdna_completion_item_kind_e {
  IREE_HAL_AMDXDNA_COMPLETION_ITEM_SUBMISSION = 0,
  IREE_HAL_AMDXDNA_COMPLETION_ITEM_ACTION = 1,
} iree_hal_amdxdna_completion_item_kind_t;

typedef struct iree_hal_amdxdna_completion_item_t {
  iree_hal_amdxdna_completion_item_kind_t kind;
  struct iree_hal_amdxdna_completion_item_t* next;
  union {
    iree_hal_amdxdna_native_submission_t* submission;
    struct {
      iree_hal_amdxdna_completion_action_fn_t fn;
      iree_hal_amdxdna_completion_cleanup_fn_t cleanup_fn;
      void* user_data;
      bool run_on_error;
    } action;
  } payload;
} iree_hal_amdxdna_completion_item_t;

typedef struct iree_hal_amdxdna_completion_cleanup_t {
  iree_hal_amdxdna_completion_cleanup_fn_t fn;
  void* user_data;
  struct iree_hal_amdxdna_completion_cleanup_t* next;
} iree_hal_amdxdna_completion_cleanup_t;

struct iree_hal_amdxdna_completion_batch_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdxdna_completion_queue_t* queue;
  iree_allocator_t host_allocator;
  iree_hal_semaphore_list_t signal_list;
  iree_status_t status;
  iree_time_t submit_time;
  iree_notification_t done_notification;
  iree_atomic_int32_t done;
  iree_atomic_int32_t submitted;
  bool cleanups_ran;
  bool native_signals_published;
  iree_hal_command_buffer_t* retained_command_buffer;
  iree_hal_amdxdna_completion_item_t* item_head;
  iree_hal_amdxdna_completion_item_t** item_tail_link;
  iree_hal_amdxdna_completion_cleanup_t* cleanup_head;
  iree_hal_amdxdna_completion_cleanup_t** cleanup_tail_link;
  struct iree_hal_amdxdna_completion_batch_t* next;
};

struct iree_hal_amdxdna_completion_queue_t {
  iree_allocator_t host_allocator;
  iree_thread_t* worker_thread;
  iree_slim_mutex_t mutex;
  iree_notification_t notification;
  iree_notification_t drain_notification;
  iree_atomic_int32_t shutdown_requested;
  iree_atomic_int32_t inflight_count;
  iree_hal_amdxdna_completion_batch_t* pending_head;
  iree_hal_amdxdna_completion_batch_t** pending_tail_link;

  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_axis_t frontier_axis;
  iree_atomic_uint64_t epoch;
  iree_atomic_uint64_t* frontier_epoch;
};

static iree_atomic_int32_t iree_hal_amdxdna_profile_completion_registered =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_finish_count =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_finish_ns =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_queue_delay_ns =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t
    iree_hal_amdxdna_profile_completion_submission_wait_ns =
        IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_action_ns =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_cleanup_ns =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_signal_ns =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_clear_ns =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_claimed_waits =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_notified_waits =
    IREE_ATOMIC_VAR_INIT(0);
static iree_atomic_uint64_t iree_hal_amdxdna_profile_completion_batch_wait_ns =
    IREE_ATOMIC_VAR_INIT(0);

static uint64_t iree_hal_amdxdna_profile_completion_load_u64(
    iree_atomic_uint64_t* value) {
  return (uint64_t)iree_atomic_load(value, iree_memory_order_relaxed);
}

static void iree_hal_amdxdna_profile_completion_dump(void) {
  fprintf(stderr,
          "[amdxdna-prof] completion finish=%" PRIu64 " claimed_waits=%" PRIu64
          " notified_waits=%" PRIu64 "\n",
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_finish_count),
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_claimed_waits),
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_notified_waits));
  fprintf(stderr,
          "[amdxdna-prof] completion_time_us finish=%" PRIu64
          " queue_delay=%" PRIu64 " submission_wait=%" PRIu64 " action=%" PRIu64
          " cleanup=%" PRIu64 " signal=%" PRIu64 " clear=%" PRIu64
          " batch_wait=%" PRIu64 "\n",
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_finish_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_queue_delay_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_submission_wait_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_action_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_cleanup_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_signal_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_clear_ns) /
              1000,
          iree_hal_amdxdna_profile_completion_load_u64(
              &iree_hal_amdxdna_profile_completion_batch_wait_ns) /
              1000);
}

static bool iree_hal_amdxdna_profile_completion_enabled(void) {
  const char* value = getenv("IREE_AMDXDNA_PROFILE_TIMING");
  if (!value || !value[0] || strcmp(value, "0") == 0) {
    value = getenv("IREE_AMDXDNA_PROFILE_CHURN");
    if (!value || !value[0] || strcmp(value, "0") == 0) return false;
  }
  int32_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &iree_hal_amdxdna_profile_completion_registered, &expected, 1,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    atexit(iree_hal_amdxdna_profile_completion_dump);
  }
  return true;
}

static int iree_hal_amdxdna_completion_worker_claim_delay_us(void) {
  const char* value = getenv("IREE_AMDXDNA_COMPLETION_WORKER_CLAIM_DELAY_US");
  if (!value || !value[0]) return 0;
  return atoi(value);
}

static void iree_hal_amdxdna_completion_worker_delay_claim(void) {
  const int delay_us = iree_hal_amdxdna_completion_worker_claim_delay_us();
  if (delay_us <= 0) return;
  const iree_time_t deadline =
      iree_time_now() + (iree_time_t)delay_us * (iree_time_t)1000;
  do {
    iree_processor_yield();
  } while (iree_time_now() < deadline);
}

static void iree_hal_amdxdna_profile_completion_inc(
    iree_atomic_uint64_t* counter) {
  if (iree_hal_amdxdna_profile_completion_enabled()) {
    iree_atomic_fetch_add(counter, 1, iree_memory_order_relaxed);
  }
}

static void iree_hal_amdxdna_profile_completion_add_ns(
    iree_atomic_uint64_t* counter, iree_time_t start_time) {
  if (iree_hal_amdxdna_profile_completion_enabled()) {
    iree_atomic_fetch_add(counter, iree_time_now() - start_time,
                          iree_memory_order_relaxed);
  }
}

static bool iree_hal_amdxdna_completion_queue_is_drained(void* arg) {
  iree_hal_amdxdna_completion_queue_t* queue =
      (iree_hal_amdxdna_completion_queue_t*)arg;
  return iree_atomic_load(&queue->inflight_count, iree_memory_order_acquire) ==
         0;
}

static bool iree_hal_amdxdna_completion_batch_is_done(void* arg) {
  iree_hal_amdxdna_completion_batch_t* batch =
      (iree_hal_amdxdna_completion_batch_t*)arg;
  return iree_atomic_load(&batch->done, iree_memory_order_acquire) != 0;
}

static void iree_hal_amdxdna_completion_batch_mark_done(
    iree_hal_amdxdna_completion_batch_t* batch) {
  iree_atomic_store(&batch->done, 1, iree_memory_order_release);
  iree_notification_post(&batch->done_notification, IREE_ALL_WAITERS);
}

static void iree_hal_amdxdna_completion_queue_complete_inflight(
    iree_hal_amdxdna_completion_queue_t* queue) {
  if (iree_atomic_fetch_sub(&queue->inflight_count, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_notification_post(&queue->drain_notification, IREE_ALL_WAITERS);
  }
}

static void iree_hal_amdxdna_completion_queue_advance_frontier(
    iree_hal_amdxdna_completion_queue_t* queue) {
  if (!queue->frontier_tracker) return;
  uint64_t epoch = (uint64_t)iree_atomic_fetch_add(queue->frontier_epoch, 1,
                                                   iree_memory_order_acq_rel) +
                   1;
  iree_async_frontier_tracker_advance(queue->frontier_tracker,
                                      queue->frontier_axis, epoch);
}

iree_hal_amdxdna_completion_batch_t* iree_hal_amdxdna_completion_batch_retain(
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (batch) iree_atomic_ref_count_inc(&batch->ref_count);
  return batch;
}

static void iree_hal_amdxdna_completion_batch_run_cleanups(
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (batch->cleanups_ran) return;
  iree_hal_amdxdna_completion_cleanup_t* cleanup = batch->cleanup_head;
  iree_hal_amdxdna_completion_cleanup_t* reversed = NULL;
  while (cleanup) {
    iree_hal_amdxdna_completion_cleanup_t* next = cleanup->next;
    cleanup->next = reversed;
    reversed = cleanup;
    cleanup = next;
  }
  batch->cleanup_head = reversed;
  batch->cleanup_tail_link = &batch->cleanup_head;
  cleanup = batch->cleanup_head;
  while (cleanup && cleanup->next) cleanup = cleanup->next;
  if (cleanup) batch->cleanup_tail_link = &cleanup->next;
  cleanup = batch->cleanup_head;
  while (cleanup) {
    if (cleanup->fn) cleanup->fn(cleanup->user_data);
    cleanup = cleanup->next;
  }
  batch->cleanups_ran = true;
}

static void iree_hal_amdxdna_completion_batch_deallocate(
    iree_hal_amdxdna_completion_batch_t* batch) {
  iree_allocator_t host_allocator = batch->host_allocator;
  iree_hal_amdxdna_completion_item_t* item = batch->item_head;
  while (item) {
    iree_hal_amdxdna_completion_item_t* next = item->next;
    if (item->kind == IREE_HAL_AMDXDNA_COMPLETION_ITEM_SUBMISSION) {
      iree_hal_amdxdna_native_submission_c_destroy(item->payload.submission);
    } else if (item->payload.action.cleanup_fn) {
      item->payload.action.cleanup_fn(item->payload.action.user_data);
    }
    iree_allocator_free(host_allocator, item);
    item = next;
  }
  iree_hal_amdxdna_completion_batch_run_cleanups(batch);
  iree_hal_amdxdna_completion_cleanup_t* cleanup = batch->cleanup_head;
  while (cleanup) {
    iree_hal_amdxdna_completion_cleanup_t* next = cleanup->next;
    iree_allocator_free(host_allocator, cleanup);
    cleanup = next;
  }
  iree_hal_command_buffer_release(batch->retained_command_buffer);
  iree_hal_semaphore_list_free(batch->signal_list, host_allocator);
  iree_status_ignore(batch->status);
  iree_notification_deinitialize(&batch->done_notification);
  iree_allocator_free(host_allocator, batch);
}

void iree_hal_amdxdna_completion_batch_destroy(
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (!batch) return;
  if (iree_atomic_ref_count_dec(&batch->ref_count) == 1) {
    iree_hal_amdxdna_completion_batch_deallocate(batch);
  }
}

static void iree_hal_amdxdna_completion_batch_publish_native_signals(
    iree_hal_amdxdna_completion_batch_t* batch) {
  for (iree_host_size_t i = 0; i < batch->signal_list.count; ++i) {
    iree_hal_amdxdna_semaphore_record_native_signal(
        batch->signal_list.semaphores[i], batch->signal_list.payload_values[i],
        batch);
  }
}

void iree_hal_amdxdna_completion_batch_publish_signals(
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (!batch || batch->native_signals_published) return;
  iree_hal_amdxdna_completion_batch_publish_native_signals(batch);
  batch->native_signals_published = true;
}

static void iree_hal_amdxdna_completion_batch_clear_native_signals(
    iree_hal_amdxdna_completion_batch_t* batch) {
  for (iree_host_size_t i = 0; i < batch->signal_list.count; ++i) {
    iree_hal_amdxdna_semaphore_clear_native_signal(
        batch->signal_list.semaphores[i], batch->signal_list.payload_values[i],
        batch);
  }
}

static void iree_hal_amdxdna_completion_batch_finish(
    iree_hal_amdxdna_completion_batch_t* batch) {
  iree_hal_amdxdna_profile_completion_inc(
      &iree_hal_amdxdna_profile_completion_finish_count);
  iree_time_t finish_start = iree_time_now();
  if (batch->submit_time) {
    iree_hal_amdxdna_profile_completion_add_ns(
        &iree_hal_amdxdna_profile_completion_queue_delay_ns,
        batch->submit_time);
  }
  iree_status_t status = iree_status_clone(batch->status);
  iree_hal_amdxdna_completion_item_t* item = batch->item_head;
  while (item) {
    if (item->kind == IREE_HAL_AMDXDNA_COMPLETION_ITEM_SUBMISSION) {
      iree_time_t submission_wait_start = iree_time_now();
      iree_status_t wait_status = iree_hal_amdxdna_native_submission_c_wait(
          item->payload.submission, UINT64_MAX);
      iree_hal_amdxdna_profile_completion_add_ns(
          &iree_hal_amdxdna_profile_completion_submission_wait_ns,
          submission_wait_start);
      if (!iree_status_is_ok(wait_status)) {
        if (iree_status_is_ok(status)) {
          status = wait_status;
        } else {
          iree_status_ignore(wait_status);
        }
      }
    } else if (item->payload.action.fn && (iree_status_is_ok(status) ||
                                           item->payload.action.run_on_error)) {
      iree_time_t action_start = iree_time_now();
      iree_status_t action_status =
          item->payload.action.fn(item->payload.action.user_data);
      iree_hal_amdxdna_profile_completion_add_ns(
          &iree_hal_amdxdna_profile_completion_action_ns, action_start);
      if (!iree_status_is_ok(action_status)) {
        if (iree_status_is_ok(status)) {
          status = action_status;
        } else {
          iree_status_ignore(action_status);
        }
      }
    }
    item = item->next;
  }

  iree_time_t cleanup_start = iree_time_now();
  iree_hal_amdxdna_completion_batch_run_cleanups(batch);
  iree_hal_amdxdna_profile_completion_add_ns(
      &iree_hal_amdxdna_profile_completion_cleanup_ns, cleanup_start);
  if (iree_status_is_ok(status)) {
    iree_time_t signal_start = iree_time_now();
    status =
        iree_hal_semaphore_list_signal(batch->signal_list, /*frontier=*/NULL);
    if (iree_status_is_ok(status)) {
      iree_hal_amdxdna_completion_queue_advance_frontier(batch->queue);
    }
    iree_hal_amdxdna_profile_completion_add_ns(
        &iree_hal_amdxdna_profile_completion_signal_ns, signal_start);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(batch->signal_list, status);
  }
  iree_time_t clear_start = iree_time_now();
  iree_hal_amdxdna_completion_batch_clear_native_signals(batch);
  iree_hal_amdxdna_profile_completion_add_ns(
      &iree_hal_amdxdna_profile_completion_clear_ns, clear_start);
  iree_hal_amdxdna_profile_completion_add_ns(
      &iree_hal_amdxdna_profile_completion_finish_ns, finish_start);
}

static bool iree_hal_amdxdna_completion_queue_pop_locked(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_hal_amdxdna_completion_batch_t** out_batch) {
  *out_batch = queue->pending_head;
  if (!*out_batch) return false;
  queue->pending_head = (*out_batch)->next;
  if (!queue->pending_head) {
    queue->pending_tail_link = &queue->pending_head;
  }
  (*out_batch)->next = NULL;
  return true;
}

static bool iree_hal_amdxdna_completion_queue_claim_locked(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_hal_amdxdna_completion_batch_t* target_batch) {
  iree_hal_amdxdna_completion_batch_t** link = &queue->pending_head;
  while (*link) {
    iree_hal_amdxdna_completion_batch_t* batch = *link;
    if (batch == target_batch) {
      *link = batch->next;
      if (!*link) queue->pending_tail_link = link;
      batch->next = NULL;
      return true;
    }
    link = &batch->next;
  }
  return false;
}

static int iree_hal_amdxdna_completion_queue_worker_main(void* arg) {
  iree_hal_amdxdna_completion_queue_t* queue =
      (iree_hal_amdxdna_completion_queue_t*)arg;
  for (;;) {
    iree_hal_amdxdna_completion_worker_delay_claim();
    iree_hal_amdxdna_completion_batch_t* batch = NULL;
    iree_slim_mutex_lock(&queue->mutex);
    iree_hal_amdxdna_completion_queue_pop_locked(queue, &batch);
    const bool shutdown = iree_atomic_load(&queue->shutdown_requested,
                                           iree_memory_order_acquire) != 0;
    iree_slim_mutex_unlock(&queue->mutex);

    if (batch) {
      iree_hal_amdxdna_completion_batch_finish(batch);
      iree_hal_amdxdna_completion_batch_mark_done(batch);
      iree_hal_amdxdna_completion_queue_complete_inflight(queue);
      iree_hal_amdxdna_completion_batch_destroy(batch);
      continue;
    }
    if (shutdown) return 0;

    iree_wait_token_t token =
        iree_notification_prepare_wait(&queue->notification);
    iree_slim_mutex_lock(&queue->mutex);
    const bool has_work = queue->pending_head != NULL;
    const bool shutdown_after_prepare =
        iree_atomic_load(&queue->shutdown_requested,
                         iree_memory_order_acquire) != 0;
    iree_slim_mutex_unlock(&queue->mutex);
    if (!has_work && !shutdown_after_prepare) {
      iree_notification_commit_wait(&queue->notification, token,
                                    /*spin_ns=*/0, IREE_TIME_INFINITE_FUTURE);
    } else {
      iree_notification_cancel_wait(&queue->notification);
    }
  }
}

iree_status_t iree_hal_amdxdna_completion_queue_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_completion_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  iree_hal_amdxdna_completion_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  memset(queue, 0, sizeof(*queue));
  queue->host_allocator = host_allocator;
  queue->pending_tail_link = &queue->pending_head;
  queue->frontier_epoch = &queue->epoch;
  iree_slim_mutex_initialize(&queue->mutex);
  iree_notification_initialize(&queue->notification);
  iree_notification_initialize(&queue->drain_notification);
  iree_atomic_store(&queue->shutdown_requested, 0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->inflight_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->epoch, (uint64_t)0, iree_memory_order_relaxed);

  iree_thread_create_params_t thread_params = {0};
  thread_params.name = iree_make_cstring_view("amdxdna-completion");
  iree_status_t status =
      iree_thread_create(iree_hal_amdxdna_completion_queue_worker_main, queue,
                         thread_params, host_allocator, &queue->worker_thread);
  if (!iree_status_is_ok(status)) {
    iree_notification_deinitialize(&queue->drain_notification);
    iree_notification_deinitialize(&queue->notification);
    iree_slim_mutex_deinitialize(&queue->mutex);
    iree_allocator_free(host_allocator, queue);
    return status;
  }

  *out_queue = queue;
  return iree_ok_status();
}

void iree_hal_amdxdna_completion_queue_set_frontier(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_atomic_uint64_t* epoch_counter) {
  if (!queue) return;
  IREE_ASSERT(tracker == NULL || queue->frontier_tracker == NULL,
              "set_frontier replacing a live frontier tracker");
  queue->frontier_tracker = tracker;
  queue->frontier_axis = axis;
  queue->frontier_epoch =
      tracker && epoch_counter ? epoch_counter : &queue->epoch;
}

void iree_hal_amdxdna_completion_queue_destroy(
    iree_hal_amdxdna_completion_queue_t* queue) {
  if (!queue) return;
  iree_atomic_store(&queue->shutdown_requested, 1, iree_memory_order_release);
  iree_notification_await(&queue->drain_notification,
                          iree_hal_amdxdna_completion_queue_is_drained, queue,
                          iree_infinite_timeout());
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  iree_thread_release(queue->worker_thread);
  iree_notification_deinitialize(&queue->drain_notification);
  iree_notification_deinitialize(&queue->notification);
  iree_slim_mutex_deinitialize(&queue->mutex);
  iree_allocator_free(queue->host_allocator, queue);
}

iree_status_t iree_hal_amdxdna_completion_batch_create(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_hal_semaphore_list_t signal_list,
    iree_hal_amdxdna_completion_batch_t** out_batch) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_batch);
  *out_batch = NULL;

  iree_hal_amdxdna_completion_batch_t* batch = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*batch), (void**)&batch));
  memset(batch, 0, sizeof(*batch));
  iree_atomic_ref_count_init(&batch->ref_count);
  batch->queue = queue;
  batch->host_allocator = queue->host_allocator;
  batch->signal_list = iree_hal_semaphore_list_empty();
  batch->status = iree_ok_status();
  iree_notification_initialize(&batch->done_notification);
  iree_atomic_store(&batch->done, 0, iree_memory_order_relaxed);
  iree_atomic_store(&batch->submitted, 0, iree_memory_order_relaxed);
  batch->item_tail_link = &batch->item_head;
  batch->cleanup_tail_link = &batch->cleanup_head;

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_list, queue->host_allocator, &batch->signal_list);
  if (!iree_status_is_ok(status)) {
    iree_notification_deinitialize(&batch->done_notification);
    iree_allocator_free(queue->host_allocator, batch);
    return status;
  }

  *out_batch = batch;
  return iree_ok_status();
}

void iree_hal_amdxdna_completion_batch_retain_command_buffer(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_command_buffer_t* command_buffer) {
  if (!batch || !command_buffer || batch->retained_command_buffer) return;
  batch->retained_command_buffer = command_buffer;
  iree_hal_command_buffer_retain(command_buffer);
}

iree_status_t iree_hal_amdxdna_completion_batch_add_submission(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_amdxdna_native_submission_t* submission) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(submission);
  iree_hal_amdxdna_completion_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(batch->host_allocator,
                                             sizeof(*item), (void**)&item));
  memset(item, 0, sizeof(*item));
  item->kind = IREE_HAL_AMDXDNA_COMPLETION_ITEM_SUBMISSION;
  item->payload.submission = submission;
  *batch->item_tail_link = item;
  batch->item_tail_link = &item->next;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_completion_batch_add_action(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_amdxdna_completion_action_fn_t action_fn,
    iree_hal_amdxdna_completion_cleanup_fn_t cleanup_fn, void* user_data,
    bool run_on_error) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(action_fn || cleanup_fn);
  iree_hal_amdxdna_completion_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(batch->host_allocator,
                                             sizeof(*item), (void**)&item));
  memset(item, 0, sizeof(*item));
  item->kind = IREE_HAL_AMDXDNA_COMPLETION_ITEM_ACTION;
  item->payload.action.fn = action_fn;
  item->payload.action.cleanup_fn = cleanup_fn;
  item->payload.action.user_data = user_data;
  item->payload.action.run_on_error = run_on_error;
  *batch->item_tail_link = item;
  batch->item_tail_link = &item->next;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_completion_batch_add_cleanup(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_amdxdna_completion_cleanup_fn_t cleanup_fn, void* user_data) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(cleanup_fn);
  iree_hal_amdxdna_completion_cleanup_t* cleanup = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      batch->host_allocator, sizeof(*cleanup), (void**)&cleanup));
  memset(cleanup, 0, sizeof(*cleanup));
  cleanup->fn = cleanup_fn;
  cleanup->user_data = user_data;
  *batch->cleanup_tail_link = cleanup;
  batch->cleanup_tail_link = &cleanup->next;
  return iree_ok_status();
}

void iree_hal_amdxdna_completion_batch_record_error(
    iree_hal_amdxdna_completion_batch_t* batch, iree_status_t status) {
  if (!batch || iree_status_is_ok(status)) return;
  if (iree_status_is_ok(batch->status)) {
    batch->status = status;
  } else {
    iree_status_ignore(status);
  }
}

bool iree_hal_amdxdna_completion_batch_has_work(
    const iree_hal_amdxdna_completion_batch_t* batch) {
  return batch && batch->item_head != NULL;
}

iree_status_t iree_hal_amdxdna_completion_batch_submit(
    iree_hal_amdxdna_completion_batch_t* batch) {
  IREE_ASSERT_ARGUMENT(batch);
  iree_hal_amdxdna_completion_queue_t* queue = batch->queue;
  if (!iree_hal_amdxdna_completion_batch_has_work(batch)) {
    if (!batch->native_signals_published) {
      iree_hal_amdxdna_completion_batch_destroy(batch);
      return iree_ok_status();
    }
  }
  if (!batch->native_signals_published) {
    iree_hal_amdxdna_completion_batch_publish_signals(batch);
  }
  if (iree_atomic_load(&batch->submitted, iree_memory_order_acquire) != 0) {
    iree_hal_amdxdna_completion_batch_destroy(batch);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "completion batch submitted more than once");
  }
  if (iree_atomic_load(&queue->shutdown_requested, iree_memory_order_acquire) !=
      0) {
    iree_hal_amdxdna_completion_batch_record_error(
        batch,
        iree_make_status(IREE_STATUS_CANCELLED, "completion queue shut down"));
    iree_hal_amdxdna_completion_batch_finish(batch);
    iree_hal_amdxdna_completion_batch_mark_done(batch);
    iree_hal_amdxdna_completion_batch_destroy(batch);
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
  batch->submit_time = iree_time_now();
  iree_atomic_store(&batch->submitted, 1, iree_memory_order_release);
  iree_atomic_fetch_add(&queue->inflight_count, 1, iree_memory_order_acq_rel);
  iree_slim_mutex_lock(&queue->mutex);
  const bool shutdown = iree_atomic_load(&queue->shutdown_requested,
                                         iree_memory_order_acquire) != 0;
  if (!shutdown) {
    *queue->pending_tail_link = batch;
    queue->pending_tail_link = &batch->next;
  }
  iree_slim_mutex_unlock(&queue->mutex);
  if (shutdown) {
    if (iree_atomic_fetch_sub(&queue->inflight_count, 1,
                              iree_memory_order_acq_rel) == 1) {
      iree_notification_post(&queue->drain_notification, IREE_ALL_WAITERS);
    }
    iree_hal_amdxdna_completion_batch_record_error(
        batch,
        iree_make_status(IREE_STATUS_CANCELLED, "completion queue shut down"));
    iree_hal_amdxdna_completion_batch_finish(batch);
    iree_hal_amdxdna_completion_batch_mark_done(batch);
    iree_hal_amdxdna_completion_batch_destroy(batch);
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  return iree_status_from_code(IREE_STATUS_DEFERRED);
}

iree_status_t iree_hal_amdxdna_completion_batch_wait(
    iree_hal_amdxdna_completion_batch_t* batch, iree_timeout_t timeout,
    iree_async_wait_flags_t flags) {
  IREE_ASSERT_ARGUMENT(batch);
  (void)flags;
  iree_time_t wait_start = iree_time_now();

  iree_hal_amdxdna_completion_queue_t* queue = batch->queue;
  bool claimed = false;
  if (iree_timeout_is_infinite(timeout) &&
      iree_atomic_load(&batch->submitted, iree_memory_order_acquire) != 0) {
    iree_slim_mutex_lock(&queue->mutex);
    claimed = iree_hal_amdxdna_completion_queue_claim_locked(queue, batch);
    iree_slim_mutex_unlock(&queue->mutex);
  }

  if (claimed) {
    iree_hal_amdxdna_profile_completion_inc(
        &iree_hal_amdxdna_profile_completion_claimed_waits);
    iree_hal_amdxdna_completion_batch_finish(batch);
    iree_hal_amdxdna_completion_batch_mark_done(batch);
    iree_hal_amdxdna_completion_queue_complete_inflight(queue);
    iree_hal_amdxdna_completion_batch_destroy(batch);
    iree_hal_amdxdna_profile_completion_add_ns(
        &iree_hal_amdxdna_profile_completion_batch_wait_ns, wait_start);
    return iree_ok_status();
  }

  if (iree_notification_await(&batch->done_notification,
                              iree_hal_amdxdna_completion_batch_is_done, batch,
                              timeout)) {
    iree_hal_amdxdna_profile_completion_inc(
        &iree_hal_amdxdna_profile_completion_notified_waits);
    iree_hal_amdxdna_profile_completion_add_ns(
        &iree_hal_amdxdna_profile_completion_batch_wait_ns, wait_start);
    return iree_ok_status();
  }
  iree_hal_amdxdna_profile_completion_add_ns(
      &iree_hal_amdxdna_profile_completion_batch_wait_ns, wait_start);
  return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}
