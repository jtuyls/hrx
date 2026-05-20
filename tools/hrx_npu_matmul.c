// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Minimal end-to-end NPU validation harness for the xrt-lite HAL driver.
//
// Drives the AMD NPU through libhrx (no IREE VM / vmfb):
//   1. HRX_GPU_DRIVER=xrt-lite -> create + enumerate the NPU device.
//   2. Load a precompiled HAL executable (amdaie-pdi-fb) from disk.
//   3. Allocate lhs (32x128 i32 = 1), rhs (128x32 i32 = 2), out (32x32 i32).
//   4. Dispatch the matmul export and read back, checking all 256 outputs
//      (each should equal 128 * 1 * 2 = 256).
//
// Usage:
//   HRX_GPU_DRIVER=xrt-lite hrx-npu-matmul <executable.amdaie-pdi-fb>
//
// Returns 0 on success (all 256 correct), non-zero otherwise.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hrx_runtime.h"

#define M 32
#define K 128
#define N 32

#define CHECK_OK(expr, label)                                                  \
  do {                                                                         \
    hrx_status_t st__ = (expr);                                                \
    if (!hrx_status_is_ok(st__)) {                                             \
      char *msg__ = NULL;                                                      \
      size_t len__ = 0;                                                        \
      hrx_status_to_string(st__, &msg__, &len__);                              \
      fprintf(stderr, "FAIL %s: %s\n", label, msg__ ? msg__ : "(no message)"); \
      hrx_status_free_message(msg__);                                          \
      hrx_status_ignore(st__);                                                 \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static int run(const char *exe_path) {
  CHECK_OK(hrx_gpu_initialize(0), "hrx_gpu_initialize");

  int device_count = 0;
  CHECK_OK(hrx_gpu_device_count(&device_count), "hrx_gpu_device_count");
  printf("xrt-lite GPU devices enumerated via libhrx: %d\n", device_count);
  if (device_count < 1) {
    fprintf(stderr, "FAIL: no NPU device enumerated\n");
    hrx_status_ignore(hrx_gpu_shutdown());
    return 3;
  }

  hrx_device_t device = NULL;
  CHECK_OK(hrx_gpu_device_get(0, &device), "hrx_gpu_device_get");
  printf("device[0] created OK\n");

  // Load the precompiled NPU HAL executable. Empty format -> inferred.
  hrx_executable_t executable = NULL;
  CHECK_OK(hrx_executable_load_file(device, exe_path, /*format=*/"amdaie-pdi-fb",
                                    &executable),
           "hrx_executable_load_file");
  // NOTE: the xrt-lite HAL executable vtable only implements .destroy; it does
  // NOT implement export_count/export_info/lookup_export_by_name (those slots
  // are NULL). Calling hrx_executable_export_count() here would dereference a
  // NULL vtable entry and crash. Like the iree-amd-aie CTS matmul_dispatch_test,
  // we dispatch entry point ordinal 0 directly.
  printf("executable loaded OK\n");

  hrx_allocator_t allocator = hrx_device_allocator(device);

  hrx_buffer_params_t params = {0};
  // DEVICE_LOCAL implies DEVICE_VISIBLE, which the xrt-lite allocator requires
  // to grant QUEUE_DISPATCH compatibility; HOST_VISIBLE lets us map for h2d/d2h.
  params.type = HRX_MEMORY_TYPE_DEVICE_LOCAL | HRX_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;
  // Allow the buffer on any queue so it is usable by queue dispatch
  // (queue_affinity=0 means "no queues" -> QUEUE_DISPATCH compatibility fails).
  params.queue_affinity = (hrx_queue_affinity_t)~0ull;

  hrx_buffer_t lhs = NULL, rhs = NULL, out = NULL;
  CHECK_OK(hrx_allocator_allocate_buffer(allocator, params,
                                         (size_t)M * K * sizeof(int32_t), &lhs),
           "allocate lhs");
  CHECK_OK(hrx_allocator_allocate_buffer(allocator, params,
                                         (size_t)K * N * sizeof(int32_t), &rhs),
           "allocate rhs");
  CHECK_OK(hrx_allocator_allocate_buffer(allocator, params,
                                         (size_t)M * N * sizeof(int32_t), &out),
           "allocate out");
  printf("buffers allocated OK\n");

  // Fill inputs: lhs = 1, rhs = 2 (expected out = K * 1 * 2 = 256).
  int32_t *lhs_data = malloc((size_t)M * K * sizeof(int32_t));
  int32_t *rhs_data = malloc((size_t)K * N * sizeof(int32_t));
  for (int i = 0; i < M * K; ++i) lhs_data[i] = 1;
  for (int i = 0; i < K * N; ++i) rhs_data[i] = 2;
  CHECK_OK(hrx_synchronous_h2d(device, lhs_data, lhs, 0,
                               (size_t)M * K * sizeof(int32_t)),
           "h2d lhs");
  CHECK_OK(hrx_synchronous_h2d(device, rhs_data, rhs, 0,
                               (size_t)K * N * sizeof(int32_t)),
           "h2d rhs");
  free(lhs_data);
  free(rhs_data);
  printf("inputs uploaded OK\n");

  hrx_dispatch_config_t config = {0};
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;

  hrx_buffer_ref_t bindings[3];
  bindings[0].buffer = lhs;
  bindings[0].offset = 0;
  bindings[0].length = (size_t)M * K * sizeof(int32_t);
  bindings[1].buffer = rhs;
  bindings[1].offset = 0;
  bindings[1].length = (size_t)K * N * sizeof(int32_t);
  bindings[2].buffer = out;
  bindings[2].offset = 0;
  bindings[2].length = (size_t)M * N * sizeof(int32_t);

  // Signal a semaphore so we can wait for the dispatch to retire.
  hrx_semaphore_t sem = NULL;
  CHECK_OK(hrx_semaphore_create(device, /*initial_value=*/0, &sem),
           "semaphore_create");
  hrx_semaphore_t sig_sems[1] = {sem};
  uint64_t sig_vals[1] = {1};
  hrx_semaphore_list_t signal_list = {sig_sems, sig_vals, 1};

  CHECK_OK(hrx_queue_dispatch(device, /*affinity=*/0,
                              /*wait=*/NULL, &signal_list, executable,
                              /*export_ordinal=*/0, &config,
                              /*constants=*/NULL, /*constants_size=*/0, bindings,
                              /*binding_count=*/3, HRX_DISPATCH_FLAG_NONE),
           "hrx_queue_dispatch");
  CHECK_OK(hrx_semaphore_wait(sem, /*value=*/1, /*timeout_ns=*/UINT64_MAX),
           "semaphore_wait");
  printf("dispatch completed OK\n");

  int32_t out_data[M * N];
  CHECK_OK(hrx_synchronous_d2h(device, out, 0, out_data,
                               (size_t)M * N * sizeof(int32_t)),
           "d2h out");

  int wrong = 0;
  const int32_t expected = (int32_t)K * 1 * 2;
  for (int i = 0; i < M * N; ++i) {
    if (out_data[i] != expected) {
      if (wrong < 8) {
        fprintf(stderr, "wrong @ %d: %d != %d\n", i, out_data[i], expected);
      }
      ++wrong;
    }
  }
  printf("output check: %d/%d correct (expected value %d)\n", M * N - wrong,
         M * N, expected);
  if (wrong != 0) {
    fprintf(stderr, "FAIL: %d/%d outputs incorrect\n", wrong, M * N);
  } else {
    printf("PASS: all %d outputs correct\n", M * N);
  }
  fflush(stdout);
  fflush(stderr);

  // HRX_SKIP_SHUTDOWN lets us isolate compute correctness from teardown: when
  // set, we leak the device/buffers and skip hrx_gpu_shutdown(). Useful while
  // an xrt-lite device-teardown heap issue is being characterized.
  const char *skip = getenv("HRX_SKIP_SHUTDOWN");
  if (skip && skip[0] && strcmp(skip, "0") != 0) {
    printf("(skipping shutdown per HRX_SKIP_SHUTDOWN)\n");
    fflush(stdout);
    _exit(wrong == 0 ? 0 : 1);
  }

  hrx_semaphore_release(sem);
  hrx_buffer_release(out);
  hrx_buffer_release(rhs);
  hrx_buffer_release(lhs);
  hrx_executable_release(executable);
  hrx_status_ignore(hrx_gpu_shutdown());

  return wrong == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <executable.amdaie-pdi-fb>\n", argv[0]);
    return 64;
  }
  return run(argv[1]);
}
