// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/completion_queue.h"

#include <atomic>
#include <chrono>
#include <thread>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct TestState {
  iree_atomic_int32_t action_count;
  iree_atomic_int32_t cleanup_count;
  int32_t cleanup_order[2];
} TestState;

typedef struct OrderedCleanup {
  TestState* state;
  int32_t value;
} OrderedCleanup;

static iree_status_t IncrementAction(void* user_data) {
  TestState* state = reinterpret_cast<TestState*>(user_data);
  iree_atomic_fetch_add(&state->action_count, 1, iree_memory_order_acq_rel);
  return iree_ok_status();
}

static void IncrementCleanup(void* user_data) {
  TestState* state = reinterpret_cast<TestState*>(user_data);
  iree_atomic_fetch_add(&state->cleanup_count, 1, iree_memory_order_acq_rel);
}

static void RecordOrderedCleanup(void* user_data) {
  OrderedCleanup* cleanup = reinterpret_cast<OrderedCleanup*>(user_data);
  int32_t index = iree_atomic_fetch_add(&cleanup->state->cleanup_count, 1,
                                        iree_memory_order_acq_rel);
  cleanup->state->cleanup_order[index] = cleanup->value;
}

static iree_async_proactor_t* CreateProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_CHECK_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  return proactor;
}

static iree_hal_semaphore_t* CreateSemaphore(iree_async_proactor_t* proactor,
                                             uint64_t initial_value) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_semaphore_create(
      proactor, IREE_HAL_QUEUE_AFFINITY_ANY, initial_value,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, iree_allocator_system(), &semaphore));
  return semaphore;
}

static iree_hal_semaphore_list_t MakeSemaphoreList(
    iree_hal_semaphore_t** semaphores, uint64_t* values,
    iree_host_size_t count) {
  iree_hal_semaphore_list_t list = iree_hal_semaphore_list_empty();
  list.count = count;
  list.semaphores = semaphores;
  list.payload_values = values;
  return list;
}

TEST(CompletionQueueTest, RunsActionAndCleanupBeforeDestroyReturns) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  TestState state = {};
  iree_atomic_store(&state.action_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&state.cleanup_count, 0, iree_memory_order_relaxed);
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_action(
      batch, IncrementAction, IncrementCleanup, &state,
      /*run_on_error=*/false));

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  EXPECT_EQ(iree_atomic_load(&state.action_count, iree_memory_order_acquire),
            1);
  EXPECT_EQ(iree_atomic_load(&state.cleanup_count, iree_memory_order_acquire),
            1);
}

TEST(CompletionQueueTest, CleanupOnlyBatchRunsCleanupWithoutWorkerSubmission) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  TestState state = {};
  iree_atomic_store(&state.cleanup_count, 0, iree_memory_order_relaxed);
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_cleanup(
      batch, IncrementCleanup, &state));

  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_submit(batch));
  EXPECT_EQ(iree_atomic_load(&state.cleanup_count, iree_memory_order_acquire),
            1);

  iree_hal_amdxdna_completion_queue_destroy(queue);
}

TEST(CompletionQueueTest, CleanupsRunInReverseRegistrationOrder) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  TestState state = {};
  iree_atomic_store(&state.cleanup_count, 0, iree_memory_order_relaxed);
  OrderedCleanup first = {&state, 1};
  OrderedCleanup second = {&state, 2};
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_cleanup(
      batch, RecordOrderedCleanup, &first));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_cleanup(
      batch, RecordOrderedCleanup, &second));

  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_submit(batch));
  EXPECT_EQ(state.cleanup_order[0], 2);
  EXPECT_EQ(state.cleanup_order[1], 1);

  iree_hal_amdxdna_completion_queue_destroy(queue);
}

TEST(CompletionQueueTest, PublishedEmptyBatchSignalsAfterSubmit) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {1};

  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, IREE_ARRAYSIZE(semaphores)),
      &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);

  std::atomic<bool> wait_started{false};
  std::atomic<iree_status_code_t> wait_code{IREE_STATUS_UNKNOWN};
  std::thread waiter([&] {
    wait_started.store(true, std::memory_order_release);
    iree_status_t status = iree_hal_semaphore_wait(
        semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    wait_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
  });

  while (!wait_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 0ull);

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  waiter.join();
  EXPECT_EQ(wait_code.load(std::memory_order_acquire), IREE_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 1ull);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, PublishedEmptyBatchFailsAfterRecordedError) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {1};

  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, IREE_ARRAYSIZE(semaphores)),
      &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);
  iree_hal_amdxdna_completion_batch_record_error(
      batch, iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected"));

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  status = iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

}  // namespace
