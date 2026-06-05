// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/native.h"

struct iree_hal_amdxdna_run_params {
  std::vector<uint32_t> control_code;
  std::vector<uint32_t> data_payload;
  // Host patch table: a flat list of (offset, arg_idx, arg_plus) triples for
  // this run, applied only by the ERT_CMD_CHAIN path.
  std::vector<uint32_t> patch_table;
};

struct iree_hal_amdxdna_kernel_params {
  std::vector<uint8_t> pdi;
  int32_t pdi_index = -1;
  std::vector<iree_hal_amdxdna_run_params> runs;
  std::string kernel_name;
  IREE_TRACE(std::string source_filename;)
  IREE_TRACE(uint32_t source_line;)
};

struct iree_hal_amdxdna_executable {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  std::vector<iree_hal_amdxdna_kernel_params> entry_points;
  // Protects the cached control-packet context below. Multiple command buffers
  // may be recorded against one executable concurrently.
  std::mutex context_mutex;
  // Shared control-packet context and CU index resolved by the PDI-carrying
  // entry point. Empty-PDI control-packet entry points reuse both. Non-control-
  // packet entry points use a fresh local context per dispatch and do not
  // mutate these fields.
  std::shared_ptr<iree_hal_amdxdna_native_context_t> context;
  iree_hal_amdxdna_native_cu_index_t context_cu_index;
  bool context_cu_index_valid = false;
};

#endif  // IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_
