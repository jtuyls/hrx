// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_CONTEXT_BLOB_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_CONTEXT_BLOB_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace iree::hal::amdxdna::mcdm {

struct ContextBlobInfo {
  std::string kernel_name;
  uint32_t column_width = 0;
  uint32_t start_column = 0;
  uint32_t pdi_count = 0;
  std::string pdi_name;
  uint64_t dpu_kernel_id = 0;
  // Base CU/kernel names from IP_LAYOUT, in CU-mask bit order. Names are
  // normalized by dropping the xclbin instance suffix after ':'.
  std::vector<std::string> kernel_names;
  std::vector<std::string> pdi_names;
  std::vector<uint64_t> dpu_kernel_ids;
};

bool BuildContextPrivateDataFromXclbin(const uint8_t* xclbin,
                                       size_t xclbin_size, uint32_t process_id,
                                       std::vector<uint8_t>* out_blob,
                                       ContextBlobInfo* out_info,
                                       std::string* out_error);

}  // namespace iree::hal::amdxdna::mcdm

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_CONTEXT_BLOB_H_
