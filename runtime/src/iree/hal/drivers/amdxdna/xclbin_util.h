// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_XCLBIN_UTIL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_XCLBIN_UTIL_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"

iree_status_t iree_hal_amdxdna_xclbin_extract_pdi(
    iree_const_byte_span_t xclbin, uint32_t pdi_index,
    std::vector<uint8_t>* out_pdi);

#endif  // IREE_HAL_DRIVERS_AMDXDNA_XCLBIN_UTIL_H_
