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

struct iree_hal_amdxdna_kernel_params {
  // Raw PDI context image from amdxdna-pdi-fb or extracted from an XADX
  // xclbin's AIE_PARTITION section for native drivers that consume PDI.
  std::vector<uint8_t> pdi;
  // AXLF/xclbin context wrapper from amdxdna-xclbin-fb for native
  // drivers that consume an xclbin-shaped context blob.
  std::vector<uint8_t> xclbin;
  std::vector<std::vector<uint32_t>> asm_inst_runlist;
  std::vector<std::vector<uint32_t>> reconf_data_runlist;
  // Host patch table parallel to `asm_inst_runlist`: each inner vector is a
  // flat list of (offset, arg_idx, arg_plus) triples for the corresponding
  // control code, applied by the ERT_CMD_CHAIN path.
  std::vector<std::vector<uint32_t>> patch_runlist;
  std::string kernel_name;
  uint32_t n_reconfigure_runs{1};
  uint32_t n_pdi_loads{1};
  // Memoized native context + CU index resolved from this entry point's
  // PDI/xclbin. Resolving goes through the device PDI context cache, which
  // copies and FNV-hashes the whole PDI on every call; caching the result per
  // entry point lets repeat dispatches skip that hash/lookup and the CU open.
  // Written under the owning executable's context_mutex; shared ownership keeps
  // the native context alive for reuse across command buffers.
  std::shared_ptr<iree_hal_amdxdna_native_context_t> cached_context;
  iree_hal_amdxdna_native_cu_index_t cached_cu_index;
  bool cached_context_valid = false;
  IREE_TRACE(std::string source_filename;)
  IREE_TRACE(uint32_t source_line;)
};

struct iree_hal_amdxdna_executable {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t entry_point_count;
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
