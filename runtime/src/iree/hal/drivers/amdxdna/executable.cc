// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/schemas/pdi_executable_def_reader.h"
#include "iree/schemas/pdi_executable_def_verifier.h"

namespace {
extern const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable;
}  // namespace

iree_string_view_t iree_hal_amdxdna_executable_format() {
  return IREE_SV("amdxdna-pdi-fb");
}

bool iree_hal_amdxdna_executable_format_supported(
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_hal_amdxdna_executable_format());
}

iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_executable, iree_hal_amdxdna_executable_vtable,
      iree_hal_amdxdna_executable);
}

static std::vector<uint32_t> iree_hal_amdxdna_uint32_vec_to_vector(
    flatbuffers_uint32_vec_t vec) {
  if (!vec) return {};
  size_t length = flatbuffers_uint32_vec_len(vec);
  if (length == 0) return {};
  return std::vector<uint32_t>(vec, vec + length);
}

static iree_status_t iree_hal_amdxdna_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_hal_amdxdna_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_amdxdna_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_amdxdna_PdiDef_vec_t pdis =
      iree_hal_amdxdna_ExecutableDef_pdis_get(executable_def);
  size_t pdi_count = iree_hal_amdxdna_PdiDef_vec_len(pdis);
  if (pdi_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no PDI present");
  }
  for (size_t i = 0; i < pdi_count; ++i) {
    iree_hal_amdxdna_PdiDef_table_t pdi =
        iree_hal_amdxdna_PdiDef_vec_at(pdis, i);
    flatbuffers_uint8_vec_t pdi_bytes = iree_hal_amdxdna_PdiDef_pdi_get(pdi);
    if (flatbuffers_uint8_vec_len(pdi_bytes) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable PDI %zu is empty", i);
    }
  }

  iree_hal_amdxdna_EntryPointDef_vec_t entry_points =
      iree_hal_amdxdna_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count =
      iree_hal_amdxdna_EntryPointDef_vec_len(entry_points);
  if (entry_point_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points found in the executable");
  }

  bool has_pdi_entry_point = false;
  for (size_t i = 0; i < entry_point_count; ++i) {
    iree_hal_amdxdna_EntryPointDef_table_t entry_point =
        iree_hal_amdxdna_EntryPointDef_vec_at(entry_points, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_EntryPointDef_name_get(entry_point);
    if (!name || flatbuffers_string_len(name) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
    int32_t pdi_index =
        iree_hal_amdxdna_EntryPointDef_pdi_index_get(entry_point);
    if (pdi_index >= 0 && static_cast<size_t>(pdi_index) >= pdi_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu pdi index %d out of range; "
                              "executable only contains %zu PDIs",
                              i, pdi_index, pdi_count);
    }
    has_pdi_entry_point |= pdi_index >= 0;

    iree_hal_amdxdna_RunDef_vec_t runs =
        iree_hal_amdxdna_EntryPointDef_runs_get(entry_point);
    size_t run_count = iree_hal_amdxdna_RunDef_vec_len(runs);
    if (run_count == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu has no runs", i);
    }
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_RunDef_table_t run =
          iree_hal_amdxdna_RunDef_vec_at(runs, run_i);
      flatbuffers_uint32_vec_t control_code =
          iree_hal_amdxdna_RunDef_control_code_get(run);
      if (flatbuffers_uint32_vec_len(control_code) == 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "entry point %zu run %zu has no control code",
                                i, run_i);
      }
      flatbuffers_uint32_vec_t patch_table =
          iree_hal_amdxdna_RunDef_patch_table_get(run);
      size_t patch_word_count = flatbuffers_uint32_vec_len(patch_table);
      if (patch_word_count % 3 != 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "entry point %zu run %zu patch_table length %zu is not a multiple "
            "of 3 (offset, arg_idx, arg_plus triples)",
            i, run_i, patch_word_count);
      }
    }
  }
  if (!has_pdi_entry_point) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry point references a PDI");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_executable_flatbuffer_verify(executable_data));

  iree_string_view_t format = iree_hal_amdxdna_executable_format();
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  (void)native_device;
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = nullptr;
  iree_hal_amdxdna_executable* executable = nullptr;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_hal_amdxdna_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_ExecutableDef_as_root(
          executable_params->executable_data.data);
  iree_hal_amdxdna_PdiDef_vec_t pdis_vec =
      iree_hal_amdxdna_ExecutableDef_pdis_get(executable_def);
  iree_hal_amdxdna_EntryPointDef_vec_t entry_points_vec =
      iree_hal_amdxdna_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      iree_hal_amdxdna_EntryPointDef_vec_len(entry_points_vec);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable),
                                reinterpret_cast<void**>(&executable)));
  // The struct holds non-trivial members (kernel_params std::vectors, the
  // context shared_ptr); placement-new it so their default constructors run
  // before any assignment. Paired with the explicit destructor call in
  // iree_hal_amdxdna_native_executable_destroy.
  new (executable) iree_hal_amdxdna_executable();

  iree_hal_resource_initialize(&iree_hal_amdxdna_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->entry_points.resize(entry_point_count);
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       ++entry_ordinal) {
    iree_hal_amdxdna_EntryPointDef_table_t entry_point_def =
        iree_hal_amdxdna_EntryPointDef_vec_at(entry_points_vec, entry_ordinal);
    iree_hal_amdxdna_kernel_params* params =
        &executable->entry_points[entry_ordinal];
    flatbuffers_string_t name =
        iree_hal_amdxdna_EntryPointDef_name_get(entry_point_def);
    params->kernel_name.assign(name, flatbuffers_string_len(name));

    int32_t pdi_index =
        iree_hal_amdxdna_EntryPointDef_pdi_index_get(entry_point_def);
    params->pdi_index = pdi_index;
    if (pdi_index >= 0) {
      iree_hal_amdxdna_PdiDef_table_t pdi_def =
          iree_hal_amdxdna_PdiDef_vec_at(pdis_vec, pdi_index);
      flatbuffers_uint8_vec_t pdi_fb = iree_hal_amdxdna_PdiDef_pdi_get(pdi_def);
      params->pdi.assign(pdi_fb, pdi_fb + flatbuffers_uint8_vec_len(pdi_fb));
    }

    iree_hal_amdxdna_RunDef_vec_t runs_vec =
        iree_hal_amdxdna_EntryPointDef_runs_get(entry_point_def);
    size_t run_count = iree_hal_amdxdna_RunDef_vec_len(runs_vec);
    params->runs.resize(run_count);
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_RunDef_table_t run_def =
          iree_hal_amdxdna_RunDef_vec_at(runs_vec, run_i);
      params->runs[run_i].control_code = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_RunDef_control_code_get(run_def));
      params->runs[run_i].data_payload = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_RunDef_data_payload_get(run_def));
      params->runs[run_i].patch_table = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_RunDef_patch_table_get(run_def));
    }

    IREE_TRACE({
      iree_hal_amdxdna_FileLineLocDef_table_t source_loc =
          iree_hal_amdxdna_EntryPointDef_source_location_get(entry_point_def);
      if (source_loc) {
        flatbuffers_string_t filename =
            iree_hal_amdxdna_FileLineLocDef_filename_get(source_loc);
        uint32_t line = iree_hal_amdxdna_FileLineLocDef_line_get(source_loc);
        if (filename) {
          params->source_filename.assign(filename,
                                         flatbuffers_string_len(filename));
        }
        params->source_line = line;
      }
    });
  }

  *out_executable = reinterpret_cast<iree_hal_executable_t*>(executable);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  // Pairs with the placement-new in iree_hal_amdxdna_native_executable_create:
  // run the destructor so non-trivial members (kernel_params std::vectors, the
  // context shared_ptr) release their allocations / drop refcounts before the
  // backing storage is freed.
  executable->~iree_hal_amdxdna_executable();
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable = {
    .destroy = iree_hal_amdxdna_native_executable_destroy,
};
}  // namespace
