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
#include "iree/hal/drivers/amdxdna/xclbin_util.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_reader.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_verifier.h"
#include "iree/schemas/pdi_executable_def_reader.h"
#include "iree/schemas/pdi_executable_def_verifier.h"

namespace {
extern const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable;

static const iree_string_view_t kAmdxdnaPdiExecutableFormat =
    iree_string_view_literal("amdxdna-pdi-fb");
static const iree_string_view_t kAmdxdnaXclbinExecutableFormat =
    iree_string_view_literal("amdxdna-xclbin-fb");
static const iree_string_view_t kAmdxdnaXclbinExecutableCompatFormat =
    iree_string_view_literal("amdaie-amdxdna-xclbin-fb");
}  // namespace

iree_string_view_t iree_hal_amdxdna_executable_format() {
  return kAmdxdnaPdiExecutableFormat;
}

bool iree_hal_amdxdna_executable_format_supported(
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                kAmdxdnaPdiExecutableFormat) ||
         iree_string_view_equal(executable_format,
                                kAmdxdnaXclbinExecutableFormat) ||
         iree_string_view_equal(executable_format,
                                kAmdxdnaXclbinExecutableCompatFormat);
}

iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_executable, iree_hal_amdxdna_executable_vtable,
      iree_hal_amdxdna_executable);
}

iree_hal_amdxdna_native_context_t*
iree_hal_amdxdna_executable_control_context_borrow(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  std::lock_guard<std::mutex> lock(executable->context_mutex);
  return executable->context.get();
}

static std::vector<uint32_t> iree_hal_amdxdna_uint32_vec_to_vector(
    flatbuffers_uint32_vec_t vec) {
  if (!vec) return {};
  size_t length = flatbuffers_uint32_vec_len(vec);
  return std::vector<uint32_t>(vec, vec + length);
}

static std::vector<uint8_t> iree_hal_amdxdna_string_to_bytes(
    flatbuffers_string_t value) {
  if (!value) return {};
  return std::vector<uint8_t>(value, value + flatbuffers_string_len(value));
}

static iree_status_t iree_hal_amdxdna_verify_run_list(
    const char* format_name, iree_host_size_t entry_index,
    iree_host_size_t run_count,
    const std::vector<std::vector<uint32_t>>& control_codes,
    const std::vector<std::vector<uint32_t>>& payloads,
    const std::vector<std::vector<uint32_t>>& patch_tables) {
  size_t payload_run_count = 0;
  for (iree_host_size_t run_i = 0; run_i < run_count; ++run_i) {
    if (control_codes[run_i].empty()) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s entry point %" PRIhsz " run %" PRIhsz
                              " has no control code",
                              format_name, entry_index, run_i);
    }
    if (patch_tables[run_i].size() % 3 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s entry point %" PRIhsz " run %" PRIhsz
                              " patch_table length %zu is not a multiple of 3",
                              format_name, entry_index, run_i,
                              patch_tables[run_i].size());
    }
    if (!payloads[run_i].empty()) {
      if ((run_i & 1) != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "%s entry point %" PRIhsz
                                " reconfiguration payload run %" PRIhsz
                                " is not in an even reconfiguration slot",
                                format_name, entry_index, run_i);
      }
      ++payload_run_count;
    }
  }
  if (payload_run_count != 0 && run_count != 2 * payload_run_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s entry point %" PRIhsz " has %" PRIhsz
        " runs but %zu reconfiguration payloads; expected paired "
        "reconfiguration/execution runs",
        format_name, entry_index, run_count, payload_run_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_pdi_flatbuffer_verify(
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
    std::vector<std::vector<uint32_t>> control_codes(run_count);
    std::vector<std::vector<uint32_t>> payloads(run_count);
    std::vector<std::vector<uint32_t>> patch_tables(run_count);
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_RunDef_table_t run =
          iree_hal_amdxdna_RunDef_vec_at(runs, run_i);
      control_codes[run_i] = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_RunDef_control_code_get(run));
      payloads[run_i] = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_RunDef_data_payload_get(run));
      patch_tables[run_i] = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_RunDef_patch_table_get(run));
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_verify_run_list(
                "PDIX", i, run_count, control_codes, payloads, patch_tables));
  }
  if (!has_pdi_entry_point) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry point references a PDI");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_xclbin_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_hal_amdxdna_xclbin_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_amdxdna_xclbin_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_xclbin_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_amdxdna_xclbin_XclbinDef_vec_t xclbins =
      iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_get(executable_def);
  size_t xclbin_count = iree_hal_amdxdna_xclbin_XclbinDef_vec_len(xclbins);
  if (xclbin_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no xclbin present");
  }
  for (size_t i = 0; i < xclbin_count; ++i) {
    iree_hal_amdxdna_xclbin_XclbinDef_table_t xclbin =
        iree_hal_amdxdna_xclbin_XclbinDef_vec_at(xclbins, i);
    flatbuffers_string_t bytes =
        iree_hal_amdxdna_xclbin_XclbinDef_xclbin_get(xclbin);
    if (!bytes || flatbuffers_string_len(bytes) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable xclbin %zu is empty", i);
    }
  }

  iree_hal_amdxdna_xclbin_EntryPointDef_vec_t entry_points =
      iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count =
      iree_hal_amdxdna_xclbin_EntryPointDef_vec_len(entry_points);
  if (entry_point_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points found in the executable");
  }

  bool has_context_entry_point = false;
  for (size_t i = 0; i < entry_point_count; ++i) {
    iree_hal_amdxdna_xclbin_EntryPointDef_table_t entry_point =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entry_points, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_xclbin_EntryPointDef_name_get(entry_point);
    if (!name || flatbuffers_string_len(name) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }

    int32_t pdi_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(entry_point);
    int32_t xclbin_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(entry_point);
    if ((pdi_index < 0) != (xclbin_index < 0)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "entry point %zu must either reference an xclbin context and PDI "
          "index or neither",
          i);
    }
    if (xclbin_index >= 0 &&
        static_cast<size_t>(xclbin_index) >= xclbin_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "entry point %zu xclbin index %d out of range; executable only "
          "contains %zu xclbins",
          i, xclbin_index, xclbin_count);
    }
    has_context_entry_point |= xclbin_index >= 0;

    iree_hal_amdxdna_xclbin_RunDef_vec_t runs =
        iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(entry_point);
    size_t run_count = iree_hal_amdxdna_xclbin_RunDef_vec_len(runs);
    if (run_count == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu has no runs", i);
    }
    std::vector<std::vector<uint32_t>> control_codes(run_count);
    std::vector<std::vector<uint32_t>> payloads(run_count);
    std::vector<std::vector<uint32_t>> patch_tables(run_count);
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_xclbin_RunDef_table_t run =
          iree_hal_amdxdna_xclbin_RunDef_vec_at(runs, run_i);
      control_codes[run_i] = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_xclbin_RunDef_control_code_get(run));
      payloads[run_i] = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_xclbin_RunDef_data_payload_get(run));
      patch_tables[run_i] = iree_hal_amdxdna_uint32_vec_to_vector(
          iree_hal_amdxdna_xclbin_RunDef_patch_table_get(run));
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_verify_run_list(
                "XADX", i, run_count, control_codes, payloads, patch_tables));
  }
  if (!has_context_entry_point) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry point references an xclbin context");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  iree_string_view_t format = kAmdxdnaPdiExecutableFormat;
  iree_status_t status =
      iree_hal_amdxdna_pdi_flatbuffer_verify(executable_data);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    status = iree_hal_amdxdna_xclbin_flatbuffer_verify(executable_data);
    if (!iree_status_is_ok(status)) return status;
    format = kAmdxdnaXclbinExecutableFormat;
  }

  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static void iree_hal_amdxdna_append_run_params(
    iree_hal_amdxdna_kernel_params* params, std::vector<uint32_t> control_code,
    std::vector<uint32_t> data_payload, std::vector<uint32_t> patch_table) {
  params->asm_inst_runlist.push_back(std::move(control_code));
  params->patch_runlist.push_back(std::move(patch_table));
  if (!data_payload.empty()) {
    params->reconf_data_runlist.push_back(std::move(data_payload));
  }
}

static iree_status_t iree_hal_amdxdna_executable_allocate(
    iree_allocator_t host_allocator, iree_host_size_t entry_point_count,
    iree_hal_amdxdna_executable** out_executable) {
  iree_hal_amdxdna_executable* executable = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*executable),
                            reinterpret_cast<void**>(&executable)));
  new (executable) iree_hal_amdxdna_executable();
  iree_hal_resource_initialize(&iree_hal_amdxdna_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->entry_point_count = entry_point_count;
  executable->entry_points.resize(entry_point_count);
  *out_executable = executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_pdi_executable_create(
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_amdxdna_pdi_flatbuffer_verify(
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

  iree_hal_amdxdna_executable* executable = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_executable_allocate(host_allocator,
                                               entry_point_count, &executable));

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
    if (pdi_index >= 0) {
      iree_hal_amdxdna_PdiDef_table_t pdi_def =
          iree_hal_amdxdna_PdiDef_vec_at(pdis_vec, pdi_index);
      flatbuffers_uint8_vec_t pdi_fb = iree_hal_amdxdna_PdiDef_pdi_get(pdi_def);
      params->pdi.assign(pdi_fb, pdi_fb + flatbuffers_uint8_vec_len(pdi_fb));
    }

    iree_hal_amdxdna_RunDef_vec_t runs_vec =
        iree_hal_amdxdna_EntryPointDef_runs_get(entry_point_def);
    size_t run_count = iree_hal_amdxdna_RunDef_vec_len(runs_vec);
    params->asm_inst_runlist.reserve(run_count);
    params->patch_runlist.reserve(run_count);
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_RunDef_table_t run_def =
          iree_hal_amdxdna_RunDef_vec_at(runs_vec, run_i);
      iree_hal_amdxdna_append_run_params(
          params,
          iree_hal_amdxdna_uint32_vec_to_vector(
              iree_hal_amdxdna_RunDef_control_code_get(run_def)),
          iree_hal_amdxdna_uint32_vec_to_vector(
              iree_hal_amdxdna_RunDef_data_payload_get(run_def)),
          iree_hal_amdxdna_uint32_vec_to_vector(
              iree_hal_amdxdna_RunDef_patch_table_get(run_def)));
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

static iree_status_t iree_hal_amdxdna_xclbin_executable_create(
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_amdxdna_xclbin_flatbuffer_verify(
                                        executable_params->executable_data));

  iree_hal_amdxdna_xclbin_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_xclbin_ExecutableDef_as_root(
          executable_params->executable_data.data);
  iree_hal_amdxdna_xclbin_XclbinDef_vec_t xclbins_vec =
      iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_get(executable_def);
  iree_hal_amdxdna_xclbin_EntryPointDef_vec_t entry_points_vec =
      iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      iree_hal_amdxdna_xclbin_EntryPointDef_vec_len(entry_points_vec);

  iree_hal_amdxdna_executable* executable = nullptr;
  iree_status_t status = iree_hal_amdxdna_executable_allocate(
      host_allocator, entry_point_count, &executable);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       ++entry_ordinal) {
    iree_hal_amdxdna_kernel_params* params =
        &executable->entry_points[entry_ordinal];
    iree_hal_amdxdna_xclbin_EntryPointDef_table_t entry_point =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entry_points_vec,
                                                     entry_ordinal);
    flatbuffers_string_t name =
        iree_hal_amdxdna_xclbin_EntryPointDef_name_get(entry_point);
    params->kernel_name.assign(name, flatbuffers_string_len(name));

    int32_t xclbin_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(entry_point);
    if (xclbin_index >= 0) {
      iree_hal_amdxdna_xclbin_XclbinDef_table_t xclbin_def =
          iree_hal_amdxdna_xclbin_XclbinDef_vec_at(xclbins_vec, xclbin_index);
      params->xclbin = iree_hal_amdxdna_string_to_bytes(
          iree_hal_amdxdna_xclbin_XclbinDef_xclbin_get(xclbin_def));
    }
    int32_t pdi_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(entry_point);
    if (pdi_index >= 0) {
      iree_byte_span_t pdi_span = iree_byte_span_empty();
      status = iree_hal_amdxdna_xclbin_extract_pdi(
          iree_make_const_byte_span(params->xclbin.data(),
                                    params->xclbin.size()),
          static_cast<uint32_t>(pdi_index), host_allocator, &pdi_span);
      if (iree_status_is_ok(status)) {
        params->pdi.assign(pdi_span.data,
                           pdi_span.data + pdi_span.data_length);
        iree_allocator_free(host_allocator, pdi_span.data);
      }
      if (!iree_status_is_ok(status)) goto fail;
    }

    iree_hal_amdxdna_xclbin_RunDef_vec_t runs =
        iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(entry_point);
    size_t run_count = iree_hal_amdxdna_xclbin_RunDef_vec_len(runs);
    params->asm_inst_runlist.reserve(run_count);
    params->patch_runlist.reserve(run_count);
    for (size_t run_ordinal = 0; run_ordinal < run_count; ++run_ordinal) {
      iree_hal_amdxdna_xclbin_RunDef_table_t run =
          iree_hal_amdxdna_xclbin_RunDef_vec_at(runs, run_ordinal);
      iree_hal_amdxdna_append_run_params(
          params,
          iree_hal_amdxdna_uint32_vec_to_vector(
              iree_hal_amdxdna_xclbin_RunDef_control_code_get(run)),
          iree_hal_amdxdna_uint32_vec_to_vector(
              iree_hal_amdxdna_xclbin_RunDef_data_payload_get(run)),
          iree_hal_amdxdna_uint32_vec_to_vector(
              iree_hal_amdxdna_xclbin_RunDef_patch_table_get(run)));
    }

    IREE_TRACE({
      iree_hal_amdxdna_xclbin_FileLineLocDef_table_t source_loc =
          iree_hal_amdxdna_xclbin_EntryPointDef_source_location_get(
              entry_point);
      if (source_loc) {
        flatbuffers_string_t filename =
            iree_hal_amdxdna_xclbin_FileLineLocDef_filename_get(source_loc);
        uint32_t line =
            iree_hal_amdxdna_xclbin_FileLineLocDef_line_get(source_loc);
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

fail:
  executable->~iree_hal_amdxdna_executable();
  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdxdna_native_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  (void)native_device;
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);

  if (iree_string_view_equal(executable_params->executable_format,
                             kAmdxdnaXclbinExecutableFormat) ||
      iree_string_view_equal(executable_params->executable_format,
                             kAmdxdnaXclbinExecutableCompatFormat)) {
    return iree_hal_amdxdna_xclbin_executable_create(
        executable_params, host_allocator, out_executable);
  }
  return iree_hal_amdxdna_pdi_executable_create(executable_params,
                                                host_allocator, out_executable);
}

static void iree_hal_amdxdna_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  executable->~iree_hal_amdxdna_executable();
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_amdxdna_native_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  return executable->entry_point_count;
}

static iree_status_t iree_hal_amdxdna_native_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna executable function out of range");
  }
  const auto& entry_point =
      executable->entry_points[iree_hal_executable_function_index(function)];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = iree_make_string_view(entry_point.kernel_name.data(),
                                         entry_point.kernel_name.size());
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)capacity;
  (void)out_parameters;
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna executable function out of range");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    const auto& entry_point = executable->entry_points[i];
    if (iree_string_view_equal(
            name, iree_make_string_view(entry_point.kernel_name.data(),
                                        entry_point.kernel_name.size()))) {
      *out_function =
          iree_hal_executable_function_from_index(static_cast<uint32_t>(i));
      return iree_ok_status();
    }
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "amdxdna executable function '%.*s' not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t iree_hal_amdxdna_native_executable_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)name;
  (void)queue_affinity;
  *out_buffer = nullptr;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "amdxdna executables do not expose globals");
}

namespace {
const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable = {
    iree_hal_amdxdna_native_executable_destroy,
    iree_hal_amdxdna_native_executable_function_count,
    iree_hal_amdxdna_native_executable_function_info,
    iree_hal_amdxdna_native_executable_function_parameters,
    iree_hal_amdxdna_native_executable_lookup_function_by_name,
    iree_hal_amdxdna_native_executable_lookup_global_by_name,
};
}  // namespace
