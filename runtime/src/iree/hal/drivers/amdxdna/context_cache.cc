// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/context_cache.h"

#include <stddef.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "iree/hal/drivers/amdxdna/device.h"

struct iree_hal_amdxdna_context_cache_key_t {
  std::vector<uint8_t> pdi;
  std::vector<uint8_t> xclbin;
  std::string kernel_name;

  bool operator==(const iree_hal_amdxdna_context_cache_key_t& rhs) const {
    return pdi == rhs.pdi && xclbin == rhs.xclbin &&
           kernel_name == rhs.kernel_name;
  }
};

struct iree_hal_amdxdna_context_cache_key_hash_t {
  size_t operator()(const iree_hal_amdxdna_context_cache_key_t& key) const {
    size_t hash = 1469598103934665603ull;
    auto mix = [&](uint8_t byte) {
      hash ^= static_cast<size_t>(byte);
      hash *= 1099511628211ull;
    };
    for (uint8_t byte : key.pdi) mix(byte);
    mix(0xff);
    for (uint8_t byte : key.xclbin) mix(byte);
    mix(0xfe);
    for (char c : key.kernel_name) mix(static_cast<uint8_t>(c));
    return hash;
  }
};

struct iree_hal_amdxdna_device_context_cache_t {
  // Keyed by the native context image selected by the DDI. Some native drivers
  // key hardware contexts by PDI + exported CU name; others accept a
  // self-describing xclbin that can expose multiple CUs.
  std::unordered_map<iree_hal_amdxdna_context_cache_key_t,
                     std::shared_ptr<iree_hal_amdxdna_native_context_t>,
                     iree_hal_amdxdna_context_cache_key_hash_t>
      contexts;
  std::mutex mutex;
};

iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create() {
  return new iree_hal_amdxdna_device_context_cache_t();
}

void iree_hal_amdxdna_device_context_cache_destroy(
    iree_hal_amdxdna_device_context_cache_t* context_cache) {
  delete context_cache;
}

void iree_hal_amdxdna_device_context_cache_clear(
    iree_hal_amdxdna_device_context_cache_t* context_cache) {
  if (!context_cache) return;
  std::lock_guard<std::mutex> lock(context_cache->mutex);
  context_cache->contexts.clear();
}

iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    std::shared_ptr<iree_hal_amdxdna_native_context_t>* out_context) {
  *out_context = nullptr;
  if (pdi.data_length == 0 && xclbin.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control-packet context cache requires context "
                            "PDI or xclbin data");
  }
  if (pdi.data_length != 0 && !pdi.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control-packet context cache PDI is NULL");
  }
  if (xclbin.data_length != 0 && !xclbin.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control-packet context cache xclbin is NULL");
  }
  if (iree_string_view_is_empty(kernel_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "control-packet context cache requires a non-empty CU name");
  }
  // Callers must pass non-empty context data (empty-PDI/xclbin entry points
  // reuse their executable's already-resolved context instead of querying the
  // cache).
  // Lock is held across create_hw_context so two threads racing on the same
  // (or different) bootstrap keys serialize on the cache; concurrent misses on
  // different PDIs are rare enough that finer-grained locking isn't worth the
  // complexity.
  iree_hal_amdxdna_context_cache_key_t key;
  std::string native_context_kernel_name(kernel_name.data, kernel_name.size);
  const bool use_xclbin_context =
      xclbin.data_length != 0 &&
      (device->native_caps.context_image_models &
       IREE_HAL_AMDXDNA_NATIVE_CONTEXT_IMAGE_MODEL_XCLBIN);
  iree_hal_amdxdna_native_context_image_t context_image;
  context_image.pdi = pdi;
  context_image.kernel_name = iree_make_string_view(
      native_context_kernel_name.data(), native_context_kernel_name.size());
  if (use_xclbin_context) {
    key.xclbin.assign(xclbin.data, xclbin.data + xclbin.data_length);
    context_image.type = iree_hal_amdxdna_native_context_image_type_t::xclbin;
    context_image.xclbin = xclbin;
  } else {
    if (pdi.data_length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "control-packet context cache requires PDI data for this native "
          "driver");
    }
    key.pdi.assign(pdi.data, pdi.data + pdi.data_length);
    key.kernel_name = native_context_kernel_name;
    context_image.type = iree_hal_amdxdna_native_context_image_type_t::pdi;
    context_image.xclbin = iree_const_byte_span_empty();
  }
  std::lock_guard<std::mutex> lock(device->context_cache->mutex);
  auto it = device->context_cache->contexts.find(key);
  if (it != device->context_cache->contexts.end()) {
    *out_context = it->second;
    return iree_ok_status();
  }
  iree_hal_amdxdna_native_context_t* raw_context = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_create_context(
      device->native_device, &context_image, &raw_context));
  std::shared_ptr<iree_hal_amdxdna_native_context_t> ctx(
      raw_context, iree_hal_amdxdna_native_context_destroy);
  device->context_cache->contexts.emplace(std::move(key), ctx);
  *out_context = ctx;
  return iree_ok_status();
}
