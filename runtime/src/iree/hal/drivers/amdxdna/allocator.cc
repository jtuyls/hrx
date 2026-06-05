// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/allocator.h"

#include <string.h>

#include <memory>

#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/util.h"

namespace {
extern const iree_hal_allocator_vtable_t iree_hal_amdxdna_allocator_vtable;
}

struct iree_hal_amdxdna_allocator {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_native_device_t* native_device;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)

  iree_hal_amdxdna_allocator(iree_allocator_t host_allocator,
                             iree_hal_amdxdna_native_device_t* native_device)
      : host_allocator(host_allocator), native_device(native_device) {
    IREE_TRACE_ZONE_BEGIN(z0);

    iree_hal_resource_initialize(&iree_hal_amdxdna_allocator_vtable,
                                 &this->resource);

    IREE_TRACE_ZONE_END(z0);
  }
};

static iree_status_t iree_hal_amdxdna_allocator_unimplemented(
    const char* operation) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "%s is not implemented",
                          operation);
}

static void iree_hal_amdxdna_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
#if IREE_STATISTICS_ENABLE
  iree_hal_amdxdna_allocator* allocator = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_allocator, iree_hal_amdxdna_allocator_vtable,
      iree_hal_amdxdna_allocator);
  *out_statistics = allocator->statistics;
#else
  (void)base_allocator;
  memset(out_statistics, 0, sizeof(*out_statistics));
#endif  // IREE_STATISTICS_ENABLE
}

static iree_status_t iree_hal_amdxdna_allocator_query_memory_heaps(
    iree_hal_allocator_t* base_allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* heaps, iree_host_size_t* out_count) {
  (void)base_allocator;
  static constexpr iree_host_size_t kHeapCount = 1;
  *out_count = kHeapCount;
  if (capacity < kHeapCount) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  heaps[0] = {
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                       IREE_HAL_BUFFER_USAGE_MAPPING,
      .max_allocation_size = IREE_DEVICE_SIZE_MAX,
      .min_alignment = 4,
  };
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_amdxdna_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  // amdxdna native host-visible allocations are mmap'd shared with the device,
  // so every allocation is simultaneously host-local, device-visible,
  // host-mappable, and host-transferable. Declaring those capabilities lets
  // iree-tooling's `requires_buffer_transfer` return false for output buffer
  // views and skip the staging copy_buffer; the host can read the dispatch
  // output buffer directly, saving one allocation + one submission + one host
  // memcpy per output.
  params->type |=
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params->usage |=
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_TRANSFER;

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer;
  // but it can happen in real world use cases. So we should at least not
  // crash.
  if (*allocation_size == 0) *allocation_size = 4;
  // Align allocation sizes to 4 bytes so shaders operating on 32 bit types
  // can act safely even on buffer ranges that are not naturally aligned.
  *allocation_size = iree_host_align(*allocation_size, 4);

  IREE_TRACE_ZONE_END(z0);
  return compatibility;
}

static iree_status_t iree_hal_amdxdna_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator* allocator = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_allocator, iree_hal_amdxdna_allocator_vtable,
      iree_hal_amdxdna_allocator);
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_amdxdna_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  iree_hal_amdxdna_native_buffer_ptr native_buffer;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdxdna_native_device_alloc_buffer(
          allocator->native_device, allocation_size,
          iree_hal_amdxdna_native_buffer_type_t::host_only, &native_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  const iree_hal_buffer_placement_t placement = {
      .queue_affinity = params->queue_affinity ? params->queue_affinity
                                               : IREE_HAL_QUEUE_AFFINITY_ANY,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_status_t status = iree_hal_amdxdna_buffer_wrap(
      native_buffer.get(), placement, compat_params.type, compat_params.access,
      compat_params.usage, allocation_size,
      /*byte_offset=*/0, /*byte_length=*/allocation_size,
      iree_hal_buffer_release_callback_null(), allocator->host_allocator,
      &buffer);

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    native_buffer.release();
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdxdna_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* base_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator* allocator = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_allocator, iree_hal_amdxdna_allocator_vtable,
      iree_hal_amdxdna_allocator);
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));
  iree_hal_buffer_destroy(base_buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdxdna_allocator_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator* allocator = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                reinterpret_cast<void**>(&allocator)));
  allocator =
      new (allocator) iree_hal_amdxdna_allocator(host_allocator, native_device);

  *out_allocator = reinterpret_cast<iree_hal_allocator_t*>(allocator);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator* allocator = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_allocator, iree_hal_amdxdna_allocator_vtable,
      iree_hal_amdxdna_allocator);
  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_amdxdna_allocator_supports_virtual_memory(
    iree_hal_allocator_t* base_allocator) {
  // XDNA exposes BOs via DRM ioctl, not a virtual-address-reservation API. A
  // real implementation would need kernel support for partial-population /
  // page-level mapping that the amdxdna driver does not provide today.
  return false;
}

static iree_allocator_t iree_hal_amdxdna_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdxdna_allocator* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           const iree_hal_amdxdna_allocator);

  IREE_TRACE_ZONE_END(z0);
  return allocator->host_allocator;
}

static iree_status_t iree_hal_amdxdna_allocator_trim(
    iree_hal_allocator_t* base_allocator) {
  (void)base_allocator;
  // BOs are owned directly by HAL buffers and released immediately on buffer
  // destruction; there is no allocator-side cache to trim.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_allocator_import_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params,
    iree_hal_external_buffer_t* external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  (void)base_allocator;
  (void)params;
  (void)external_buffer;
  (void)release_callback;
  (void)out_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("buffer import");
}

static iree_status_t iree_hal_amdxdna_allocator_export_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer) {
  (void)base_allocator;
  (void)buffer;
  (void)requested_type;
  (void)requested_flags;
  (void)out_external_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("buffer export");
}

static iree_status_t
iree_hal_amdxdna_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size) {
  (void)base_allocator;
  (void)params;
  (void)out_minimum_page_size;
  (void)out_recommended_page_size;
  return iree_hal_amdxdna_allocator_unimplemented(
      "virtual memory granularity query");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** out_virtual_buffer) {
  (void)base_allocator;
  (void)queue_affinity;
  (void)size;
  (void)out_virtual_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory reserve");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_release(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer) {
  (void)base_allocator;
  (void)virtual_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory release");
}

static iree_status_t iree_hal_amdxdna_allocator_physical_memory_allocate(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  (void)base_allocator;
  (void)params;
  (void)size;
  (void)host_allocator;
  (void)out_physical_memory;
  return iree_hal_amdxdna_allocator_unimplemented("physical memory allocate");
}

static iree_status_t iree_hal_amdxdna_allocator_physical_memory_free(
    iree_hal_allocator_t* base_allocator,
    iree_hal_physical_memory_t* physical_memory) {
  (void)base_allocator;
  (void)physical_memory;
  return iree_hal_amdxdna_allocator_unimplemented("physical memory free");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_map(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)physical_memory;
  (void)physical_offset;
  (void)size;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory map");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory unmap");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_protect(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)protection;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory protect");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_advise(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)advice;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory advise");
}

namespace {
const iree_hal_allocator_vtable_t iree_hal_amdxdna_allocator_vtable = {
    .destroy = iree_hal_amdxdna_allocator_destroy,
    .host_allocator = iree_hal_amdxdna_allocator_host_allocator,
    .trim = iree_hal_amdxdna_allocator_trim,
    .query_statistics = iree_hal_amdxdna_allocator_query_statistics,
    .query_memory_heaps = iree_hal_amdxdna_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_amdxdna_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_amdxdna_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_amdxdna_allocator_deallocate_buffer,
    // Stubs that return UNIMPLEMENTED. The XDNA kernel ABI has no
    // wrap-host-pointer / userptr ioctl (no equivalent of
    // hipHostRegister/cuMemHostRegister), so a real zero-copy import is not
    // possible without kernel changes. Today the AllocatorTest.Import*
    // CTS tests SKIP at compatibility-check time because
    // query_buffer_compatibility above never sets IMPORTABLE; callers that
    // do reach iree_hal_memory_file_wrap see UNIMPLEMENTED here and fall
    // back to a HOST_LOCAL | HOST_COHERENT heap buffer; fine for transfer
    // source/target use but not DEVICE_VISIBLE so it cannot be a dispatch
    // binding. A real implementation would have to allocate a SHMEM BO and
    // memcpy host data into it (gives a dispatch-capable buffer at the cost
    // of one copy), or add a kernel userptr path (zero-copy, kernel work).
    .import_buffer = iree_hal_amdxdna_allocator_import_buffer,
    .export_buffer = iree_hal_amdxdna_allocator_export_buffer,
    // Virtual memory is not supported on XDNA. supports_virtual_memory
    // returns false so callers short-circuit before reaching the rest of
    // the VM vtable; the remaining slots are filled with UNIMPLEMENTED
    // stubs to keep the vtable complete (NULL fn pointers would SEGV any
    // caller that bypasses the supports check).
    .supports_virtual_memory =
        iree_hal_amdxdna_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        iree_hal_amdxdna_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve = iree_hal_amdxdna_allocator_virtual_memory_reserve,
    .virtual_memory_release = iree_hal_amdxdna_allocator_virtual_memory_release,
    .physical_memory_allocate =
        iree_hal_amdxdna_allocator_physical_memory_allocate,
    .physical_memory_free = iree_hal_amdxdna_allocator_physical_memory_free,
    .virtual_memory_map = iree_hal_amdxdna_allocator_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_amdxdna_allocator_virtual_memory_unmap,
    .virtual_memory_protect = iree_hal_amdxdna_allocator_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_amdxdna_allocator_virtual_memory_advise,
};
}
