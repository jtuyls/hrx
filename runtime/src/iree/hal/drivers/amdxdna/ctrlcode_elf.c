// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/ctrlcode_elf.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  IREE_HAL_AMDXDNA_ELF32_CLASS = 1,
  IREE_HAL_AMDXDNA_ELF32_DATA_LE = 1,
  IREE_HAL_AMDXDNA_ELF32_VERSION = 1,
  IREE_HAL_AMDXDNA_ELF32_EHDR_SIZE = 52,
  IREE_HAL_AMDXDNA_ELF32_SHDR_SIZE = 40,
  IREE_HAL_AMDXDNA_ELF32_SYM_SIZE = 16,
  IREE_HAL_AMDXDNA_ELF32_RELA_SIZE = 12,
  IREE_HAL_AMDXDNA_ELF_EI_CLASS = 4,
  IREE_HAL_AMDXDNA_ELF_EI_DATA = 5,
  IREE_HAL_AMDXDNA_ELF_EI_VERSION = 6,
  IREE_HAL_AMDXDNA_ELF_EI_OSABI = 7,
  IREE_HAL_AMDXDNA_ELF_EI_ABIVERSION = 8,
  IREE_HAL_AMDXDNA_ELF32_E_SHOFF = 0x20,
  IREE_HAL_AMDXDNA_ELF32_E_SHENTSIZE = 0x2E,
  IREE_HAL_AMDXDNA_ELF32_E_SHNUM = 0x30,
  IREE_HAL_AMDXDNA_ELF32_E_SHSTRNDX = 0x32,
  IREE_HAL_AMDXDNA_ELF32_SH_NAME = 0,
  IREE_HAL_AMDXDNA_ELF32_SH_OFFSET = 0x10,
  IREE_HAL_AMDXDNA_ELF32_SH_SIZE = 0x14,
  IREE_HAL_AMDXDNA_ELF32_SH_LINK = 0x18,
  IREE_HAL_AMDXDNA_ELF32_SH_ENTSIZE = 0x24,
  IREE_HAL_AMDXDNA_ELF32_ST_NAME = 0,
  IREE_HAL_AMDXDNA_ELF32_ST_SHNDX = 14,
  IREE_HAL_AMDXDNA_CTRLCODE_PAGE_SIZE = 8192,
  IREE_HAL_AMDXDNA_CTRLCODE_PAGE_HDR = 16,
  IREE_HAL_AMDXDNA_CTRLCODE_ADDEND_SHIFT = 4,
  // npu3 validate gemm.elf has 260 sections (.ctrltext/.ctrldata per page).
  IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS = 512,
  IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS = 16,
  // KMD fill_indirect_pkt rejects uc_index or chained >= this
  // (HSA_MAX_LEVEL1_INDIRECT_ENTRIES). chained on the first entry is count-1,
  // so six occupied columns is the maximum indirect packet.
  IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES = 6,
};

typedef struct iree_hal_amdxdna_ctrl_page_t {
  uint32_t col;
  uint32_t page;
  uint32_t ctrltext_offset;
  uint32_t ctrltext_size;
  uint32_t ctrldata_offset;
  uint32_t ctrldata_size;
  bool has_ctrltext;
  bool has_ctrldata;
} iree_hal_amdxdna_ctrl_page_t;

typedef struct iree_hal_amdxdna_elf_section_t {
  uint32_t name_off;
  uint32_t file_offset;
  uint32_t size;
  uint32_t link;
  uint32_t entsize;
  uint32_t index;
} iree_hal_amdxdna_elf_section_t;

void iree_hal_amdxdna_ctrlcode_elf_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_ctrlcode_elf_t* elf) {
  if (!elf) return;
  iree_allocator_free(host_allocator, elf->ctrl_words);
  iree_allocator_free(host_allocator, elf->patches);
  iree_allocator_free(host_allocator, elf->dpu_slices);
  memset(elf, 0, sizeof(*elf));
}

static bool iree_hal_amdxdna_elf_range_in_bounds(size_t data_size,
                                                 uint32_t offset,
                                                 uint32_t size) {
  return (size_t)offset <= data_size &&
         (size_t)size <= data_size - (size_t)offset;
}

static uint16_t iree_hal_amdxdna_read_u16(const uint8_t* data, size_t offset) {
  uint16_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static uint32_t iree_hal_amdxdna_read_u32(const uint8_t* data, size_t offset) {
  uint32_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static int32_t iree_hal_amdxdna_read_i32(const uint8_t* data, size_t offset) {
  int32_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static iree_status_t iree_hal_amdxdna_elf_section_name(
    iree_const_byte_span_t elf, const iree_hal_amdxdna_elf_section_t* shstr,
    const iree_hal_amdxdna_elf_section_t* section, iree_string_view_t* out) {
  *out = iree_string_view_empty();
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, shstr->file_offset,
                                            shstr->size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF shstrtab is out of range");
  }
  if (section->name_off >= shstr->size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section name is out of range");
  }
  const char* start =
      (const char*)elf.data + shstr->file_offset + section->name_off;
  size_t max_len = shstr->size - section->name_off;
  size_t len = 0;
  while (len < max_len && start[len] != '\0') ++len;
  if (len == max_len) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section name is not "
                            "NUL-terminated");
  }
  *out = iree_make_string_view(start, len);
  return iree_ok_status();
}

static bool iree_hal_amdxdna_name_is_ctrltext(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".ctrltext")) ||
         iree_string_view_starts_with(name, IREE_SV(".ctrltext."));
}

static bool iree_hal_amdxdna_name_is_ctrldata(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".ctrldata")) ||
         iree_string_view_starts_with(name, IREE_SV(".ctrldata."));
}

static bool iree_hal_amdxdna_name_is_rela_dyn(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".rela.dyn")) ||
         iree_string_view_starts_with(name, IREE_SV(".rela.dyn."));
}

static bool iree_hal_amdxdna_name_is_pad(iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".pad")) ||
         iree_string_view_starts_with(name, IREE_SV(".pad."));
}

static bool iree_hal_amdxdna_name_is_xrt_configuration(
    iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV(".note.xrt.configuration"));
}

// ELF32 note: namesz/descsz/type, name padded to 4, then desc. XRT stores the
// partition column count as the first little-endian u32 of the descriptor.
static uint32_t iree_hal_amdxdna_parse_partition_cols(
    iree_const_byte_span_t elf, const iree_hal_amdxdna_elf_section_t* note) {
  if (!note || note->size < 16) return 0;
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, note->file_offset,
                                            note->size)) {
    return 0;
  }
  const uint8_t* blob = elf.data + note->file_offset;
  uint32_t namesz = iree_hal_amdxdna_read_u32(blob, 0);
  uint32_t descsz = iree_hal_amdxdna_read_u32(blob, 4);
  uint32_t name_pad = (namesz + 3u) & ~3u;
  if (descsz < 4 || (uint64_t)12 + name_pad + 4 > note->size) return 0;
  return iree_hal_amdxdna_read_u32(blob, 12 + name_pad);
}

static bool iree_hal_amdxdna_parse_u32_token(const char* start, const char* end,
                                             uint32_t* out_value,
                                             const char** out_next) {
  if (start >= end || *start < '0' || *start > '9') return false;
  uint32_t value = 0;
  const char* p = start;
  while (p < end && *p >= '0' && *p <= '9') {
    uint32_t digit = (uint32_t)(*p - '0');
    if (value > (UINT32_MAX - digit) / 10) return false;
    value = value * 10 + digit;
    ++p;
  }
  *out_value = value;
  *out_next = p;
  return true;
}

static iree_status_t iree_hal_amdxdna_parse_col_page(iree_string_view_t name,
                                                     uint32_t* out_col,
                                                     uint32_t* out_page) {
  *out_col = 0;
  *out_page = 0;
  // ".ctrltext" and ".ctrldata" are both 9 characters; extra ".col.page"
  // components follow that prefix.
  if (name.size <= 9) return iree_ok_status();
  if (name.data[9] != '.') return iree_ok_status();
  const char* p = name.data + 10;
  const char* end = name.data + name.size;
  uint32_t col = 0;
  if (!iree_hal_amdxdna_parse_u32_token(p, end, &col, &p)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section '%.*s' has an "
                            "invalid column index",
                            (int)name.size, name.data);
  }
  uint32_t page = 0;
  if (p < end) {
    if (*p != '.') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page suffix",
                              (int)name.size, name.data);
    }
    ++p;
    if (!iree_hal_amdxdna_parse_u32_token(p, end, &page, &p)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page index",
                              (int)name.size, name.data);
    }
  }
  // AIE4 OSABI 70 uses `.ctrltext.<col>.<page>.<uc>` (and the same for
  // `.ctrldata`). Ignore a trailing numeric component.
  if (p < end) {
    if (*p != '.') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page index",
                              (int)name.size, name.data);
    }
    ++p;
    uint32_t ignored = 0;
    if (!iree_hal_amdxdna_parse_u32_token(p, end, &ignored, &p) || p != end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section '%.*s' has an "
                              "invalid page index",
                              (int)name.size, name.data);
    }
  }
  *out_col = col;
  *out_page = page;
  return iree_ok_status();
}

static iree_hal_amdxdna_ctrl_page_t* iree_hal_amdxdna_find_or_add_page(
    iree_hal_amdxdna_ctrl_page_t* pages, iree_host_size_t* page_count,
    uint32_t col, uint32_t page) {
  for (iree_host_size_t i = 0; i < *page_count; ++i) {
    if (pages[i].col == col && pages[i].page == page) return &pages[i];
  }
  if (*page_count >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS) return NULL;
  iree_hal_amdxdna_ctrl_page_t* slot = &pages[*page_count];
  memset(slot, 0, sizeof(*slot));
  slot->col = col;
  slot->page = page;
  ++(*page_count);
  return slot;
}

static int iree_hal_amdxdna_ctrl_page_cmp(const void* lhs, const void* rhs) {
  const iree_hal_amdxdna_ctrl_page_t* a =
      (const iree_hal_amdxdna_ctrl_page_t*)lhs;
  const iree_hal_amdxdna_ctrl_page_t* b =
      (const iree_hal_amdxdna_ctrl_page_t*)rhs;
  if (a->col != b->col) return a->col < b->col ? -1 : 1;
  if (a->page != b->page) return a->page < b->page ? -1 : 1;
  return 0;
}

static iree_status_t iree_hal_amdxdna_append_bytes(
    iree_allocator_t host_allocator, uint8_t** io_data, size_t* io_size,
    size_t* io_capacity, const uint8_t* src, size_t src_size) {
  if (src_size == 0) return iree_ok_status();
  size_t needed = *io_size + src_size;
  if (needed < *io_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna ctrlcode ELF buffer overflow");
  }
  if (needed > *io_capacity) {
    size_t new_capacity = *io_capacity ? *io_capacity : 256;
    while (new_capacity < needed) {
      if (new_capacity > (SIZE_MAX / 2)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "amdxdna ctrlcode ELF buffer overflow");
      }
      new_capacity *= 2;
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(host_allocator, new_capacity,
                                                (void**)io_data));
    *io_capacity = new_capacity;
  }
  memcpy(*io_data + *io_size, src, src_size);
  *io_size = needed;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_pad_to_page(iree_allocator_t host_allocator,
                                                  uint8_t** io_data,
                                                  size_t* io_size,
                                                  size_t* io_capacity,
                                                  size_t col_base,
                                                  uint32_t page) {
  // Page indices restart at 0 in each column. Pad relative to that column's
  // concatenated start, not the whole instruction blob (AIE4 nop.elf is
  // `.ctrltext.{0,2,4}.0`, each an 8KiB page 0).
  const size_t pad =
      col_base + ((size_t)page + 1) * IREE_HAL_AMDXDNA_CTRLCODE_PAGE_SIZE;
  if (*io_size > pad) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF page %u exceeds the %zu-byte "
                            "page size",
                            page, pad);
  }
  if (pad == *io_size) return iree_ok_status();
  size_t old_size = *io_size;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_append_bytes(
      host_allocator, io_data, io_size, io_capacity, NULL, 0));
  if (pad > *io_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_realloc(host_allocator, pad, (void**)io_data));
    *io_capacity = pad;
  }
  memset(*io_data + old_size, 0, pad - old_size);
  *io_size = pad;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_section_bytes(
    iree_const_byte_span_t elf, uint32_t file_offset, uint32_t size,
    iree_allocator_t host_allocator, uint8_t** io_data, size_t* io_size,
    size_t* io_capacity) {
  if (size == 0) return iree_ok_status();
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, file_offset,
                                            size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section bytes are out of "
                            "range");
  }
  return iree_hal_amdxdna_append_bytes(host_allocator, io_data, io_size,
                                       io_capacity, elf.data + file_offset,
                                       size);
}

static bool iree_hal_amdxdna_is_skipped_symbol(iree_string_view_t name) {
  if (iree_string_view_is_empty(name)) return true;
  if (iree_string_view_equal(name, IREE_SV("scratch-pad-mem"))) return true;
  if (iree_string_view_equal(name, IREE_SV("scratch-pad-ctrl"))) return true;
  if (iree_string_view_equal(name, IREE_SV("control-packet"))) return true;
  if (iree_string_view_starts_with(name, IREE_SV("ctrlpkt-pm"))) return true;
  if (iree_string_view_find(name, IREE_SV("pdi"), 0) != IREE_STRING_VIEW_NPOS) {
    return true;
  }
  return false;
}

static bool iree_hal_amdxdna_parse_arg_index(iree_string_view_t name,
                                             uint32_t* out_index) {
  iree_string_view_t token = name;
  if (iree_string_view_starts_with(name, IREE_SV("argv"))) {
    token = iree_string_view_remove_prefix(name, 4);
  }
  if (iree_string_view_is_empty(token)) return false;
  for (iree_host_size_t i = 0; i < token.size; ++i) {
    if (token.data[i] < '0' || token.data[i] > '9') return false;
  }
  uint32_t value = 0;
  const char* next = NULL;
  if (!iree_hal_amdxdna_parse_u32_token(token.data, token.data + token.size,
                                        &value, &next) ||
      next != token.data + token.size) {
    return false;
  }
  *out_index = value;
  return true;
}

static iree_status_t iree_hal_amdxdna_append_patch(
    iree_allocator_t host_allocator, uint32_t** io_patches, size_t* io_count,
    size_t* io_capacity, uint32_t offset, uint32_t arg_idx, uint32_t arg_plus) {
  size_t needed = *io_count + 3;
  if (needed > *io_capacity) {
    size_t new_capacity = *io_capacity ? *io_capacity * 2 : 24;
    while (new_capacity < needed) new_capacity *= 2;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        host_allocator, new_capacity * sizeof(uint32_t), (void**)io_patches));
    *io_capacity = new_capacity;
  }
  (*io_patches)[*io_count + 0] = offset;
  (*io_patches)[*io_count + 1] = arg_idx;
  (*io_patches)[*io_count + 2] = arg_plus;
  *io_count = needed;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_load_sections(
    iree_const_byte_span_t elf, iree_hal_amdxdna_elf_section_t* sections,
    uint16_t shnum, uint32_t shoff, uint16_t shentsize) {
  for (uint16_t i = 0; i < shnum; ++i) {
    size_t sh_off = (size_t)shoff + (size_t)i * shentsize;
    if (sh_off + IREE_HAL_AMDXDNA_ELF32_SHDR_SIZE > elf.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF section header %u is out "
                              "of range",
                              i);
    }
    sections[i].name_off =
        iree_hal_amdxdna_read_u32(elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_NAME);
    sections[i].file_offset = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_OFFSET);
    sections[i].size = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_SIZE);
    sections[i].link = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_LINK);
    sections[i].entsize = iree_hal_amdxdna_read_u32(
        elf.data, sh_off + IREE_HAL_AMDXDNA_ELF32_SH_ENTSIZE);
    sections[i].index = i;
  }
  return iree_ok_status();
}

typedef struct iree_hal_amdxdna_pad_span_t {
  uint32_t col;
  uint32_t file_offset;
  uint32_t size;
} iree_hal_amdxdna_pad_span_t;

// `.pad`, `.pad.<col>`, or `.pad.<col>.<page>`. Bare `.pad` is column 0.
static iree_status_t iree_hal_amdxdna_parse_pad_col(iree_string_view_t name,
                                                    uint32_t* out_col) {
  *out_col = 0;
  if (iree_string_view_equal(name, IREE_SV(".pad"))) return iree_ok_status();
  if (!iree_string_view_starts_with(name, IREE_SV(".pad."))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section '%.*s' is not a pad",
                            (int)name.size, name.data);
  }
  const char* p = name.data + 5;
  const char* end = name.data + name.size;
  if (!iree_hal_amdxdna_parse_u32_token(p, end, out_col, &p)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section '%.*s' has an "
                            "invalid column index",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_build_aie2ps_ctrl(
    iree_const_byte_span_t elf, iree_allocator_t host_allocator,
    const iree_hal_amdxdna_ctrl_page_t* pages, iree_host_size_t page_count,
    const iree_hal_amdxdna_pad_span_t* pads, iree_host_size_t pad_count,
    uint8_t** out_data, size_t* out_size, uint32_t* out_col_bases,
    uint32_t* out_col_count) {
  *out_data = NULL;
  *out_size = 0;
  *out_col_count = 0;
  memset(out_col_bases, 0,
         IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS * sizeof(uint32_t));
  if (page_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF has no .ctrltext section");
  }

  uint32_t max_col = 0;
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    if (pages[i].col >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF column %u exceeds %u",
                              pages[i].col,
                              IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS);
    }
    if (pages[i].col > max_col) max_col = pages[i].col;
  }
  for (iree_host_size_t i = 0; i < pad_count; ++i) {
    if (pads[i].col > max_col) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF pad column %u is outside "
                              "ctrltext columns 0..%u",
                              pads[i].col, max_col);
    }
  }
  *out_col_count = max_col + 1;

  uint8_t* data = NULL;
  size_t size = 0;
  size_t capacity = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t col = 0; col <= max_col && iree_status_is_ok(status); ++col) {
    out_col_bases[col] = (uint32_t)size;
    for (iree_host_size_t i = 0; i < page_count && iree_status_is_ok(status);
         ++i) {
      if (pages[i].col != col) continue;
      if (pages[i].has_ctrltext) {
        status = iree_hal_amdxdna_copy_section_bytes(
            elf, pages[i].ctrltext_offset, pages[i].ctrltext_size,
            host_allocator, &data, &size, &capacity);
      }
      if (iree_status_is_ok(status) && pages[i].has_ctrldata) {
        status = iree_hal_amdxdna_copy_section_bytes(
            elf, pages[i].ctrldata_offset, pages[i].ctrldata_size,
            host_allocator, &data, &size, &capacity);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_pad_to_page(
            host_allocator, &data, &size, &capacity, out_col_bases[col],
            pages[i].page);
      }
    }
    // aiebu appends this column's `.pad.*` before the next column starts.
    for (iree_host_size_t p = 0; p < pad_count && iree_status_is_ok(status);
         ++p) {
      if (pads[p].col != col) continue;
      status = iree_hal_amdxdna_copy_section_bytes(
          elf, pads[p].file_offset, pads[p].size, host_allocator, &data, &size,
          &capacity);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, data);
    return status;
  }
  if (size % sizeof(uint32_t) != 0) {
    iree_allocator_free(host_allocator, data);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF instruction size %zu is not "
                            "a multiple of 4",
                            size);
  }
  *out_data = data;
  *out_size = size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_parse_relocs(
    iree_const_byte_span_t elf, uint8_t abi_version, bool paged,
    const iree_hal_amdxdna_elf_section_t* sections, uint16_t shnum,
    const iree_hal_amdxdna_elf_section_t* shstr,
    const iree_hal_amdxdna_elf_section_t* dynsym,
    const iree_hal_amdxdna_elf_section_t* dynstr,
    const uint32_t* col_bases, uint32_t col_count, size_t ctrl_size,
    iree_allocator_t host_allocator, uint32_t** out_patches,
    size_t* out_patch_count, iree_host_size_t* out_skipped) {
  *out_patches = NULL;
  *out_patch_count = 0;
  *out_skipped = 0;
  if (!dynsym || !dynstr) return iree_ok_status();
  uint32_t symentsz = dynsym->entsize ? dynsym->entsize
                                      : IREE_HAL_AMDXDNA_ELF32_SYM_SIZE;
  if (symentsz < IREE_HAL_AMDXDNA_ELF32_SYM_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF dynsym entsize %u is invalid",
                            symentsz);
  }
  if (!iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, dynsym->file_offset,
                                            dynsym->size) ||
      !iree_hal_amdxdna_elf_range_in_bounds(elf.data_length, dynstr->file_offset,
                                            dynstr->size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF dynsym/dynstr is out of range");
  }

  uint32_t* patches = NULL;
  size_t patch_count = 0;
  size_t patch_capacity = 0;
  iree_host_size_t skipped = 0;
  iree_status_t status = iree_ok_status();

  for (uint16_t si = 0; si < shnum && iree_status_is_ok(status); ++si) {
    iree_string_view_t sec_name = iree_string_view_empty();
    status = iree_hal_amdxdna_elf_section_name(elf, shstr, &sections[si],
                                               &sec_name);
    if (!iree_status_is_ok(status)) break;
    if (!iree_hal_amdxdna_name_is_rela_dyn(sec_name)) continue;
    if (sections[si].size % IREE_HAL_AMDXDNA_ELF32_RELA_SIZE != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "amdxdna ctrlcode ELF .rela.dyn size is not a "
                                "multiple of 12");
      break;
    }
    if (!iree_hal_amdxdna_elf_range_in_bounds(
            elf.data_length, sections[si].file_offset, sections[si].size)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "amdxdna ctrlcode ELF .rela.dyn is out of range");
      break;
    }
    const uint32_t rela_count =
        sections[si].size / IREE_HAL_AMDXDNA_ELF32_RELA_SIZE;
    for (uint32_t ri = 0; ri < rela_count && iree_status_is_ok(status); ++ri) {
      size_t rela_off = (size_t)sections[si].file_offset +
                        (size_t)ri * IREE_HAL_AMDXDNA_ELF32_RELA_SIZE;
      uint32_t r_offset = iree_hal_amdxdna_read_u32(elf.data, rela_off);
      uint32_t r_info = iree_hal_amdxdna_read_u32(elf.data, rela_off + 4);
      int32_t r_addend = iree_hal_amdxdna_read_i32(elf.data, rela_off + 8);
      uint32_t symidx = r_info >> 8;
      uint32_t r_type = r_info & 0xFFu;
      uint64_t sym_off = (uint64_t)symidx * symentsz;
      if (sym_off + IREE_HAL_AMDXDNA_ELF32_SYM_SIZE > dynsym->size) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "amdxdna ctrlcode ELF reloc symbol index %u "
                                  "is out of range",
                                  symidx);
        break;
      }
      uint32_t st_name = iree_hal_amdxdna_read_u32(
          elf.data, dynsym->file_offset + (size_t)sym_off +
                        IREE_HAL_AMDXDNA_ELF32_ST_NAME);
      uint16_t st_shndx = iree_hal_amdxdna_read_u16(
          elf.data, dynsym->file_offset + (size_t)sym_off +
                        IREE_HAL_AMDXDNA_ELF32_ST_SHNDX);
      if (st_name >= dynstr->size) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "amdxdna ctrlcode ELF symbol name is out of "
                                  "range");
        break;
      }
      const char* sym_start =
          (const char*)elf.data + dynstr->file_offset + st_name;
      size_t max_len = dynstr->size - st_name;
      size_t name_len = 0;
      while (name_len < max_len && sym_start[name_len] != '\0') ++name_len;
      iree_string_view_t symname =
          iree_make_string_view(sym_start, name_len);

      uint32_t scheme = r_type;
      uint32_t add_end_addr = (uint32_t)r_addend;
      if (abi_version == 1) {
        add_end_addr = ((uint32_t)r_addend) >> IREE_HAL_AMDXDNA_CTRLCODE_ADDEND_SHIFT;
        scheme = ((uint32_t)r_addend) & 0xFu;
      }

      const bool is_control_code =
          iree_string_view_starts_with(symname, IREE_SV("control-code"));
      if ((!is_control_code &&
           iree_hal_amdxdna_is_skipped_symbol(symname)) ||
          scheme != IREE_HAL_AMDXDNA_CTRLCODE_ELF_PATCH_SHIM_DMA_AIE4) {
        ++skipped;
        continue;
      }
      uint32_t arg_idx = 0;
      if (is_control_code) {
        arg_idx = UINT32_MAX;
      } else if (!iree_hal_amdxdna_parse_arg_index(symname, &arg_idx)) {
        ++skipped;
        continue;
      }

      uint32_t abs_offset = r_offset;
      if (paged) {
        if (st_shndx >= shnum) {
          ++skipped;
          continue;
        }
        iree_string_view_t patch_sec_name = iree_string_view_empty();
        status = iree_hal_amdxdna_elf_section_name(elf, shstr, &sections[st_shndx],
                                                   &patch_sec_name);
        if (!iree_status_is_ok(status)) break;
        uint32_t col = 0;
        uint32_t page = 0;
        status = iree_hal_amdxdna_parse_col_page(patch_sec_name, &col, &page);
        if (!iree_status_is_ok(status)) break;
        if (col >= col_count) {
          ++skipped;
          continue;
        }
        // st_shndx names the page; r_offset is relative to the 16-byte page
        // header (same AIE2PS/OSABI-70 convention XRT/aiebu uses). Packed
        // layout is [ctrltext][ctrldata] zero-padded to 8KiB.
        uint64_t sec_offset = (uint64_t)page * IREE_HAL_AMDXDNA_CTRLCODE_PAGE_SIZE +
                              r_offset + IREE_HAL_AMDXDNA_CTRLCODE_PAGE_HDR;
        uint64_t total = (uint64_t)col_bases[col] + sec_offset;
        if (total > UINT32_MAX) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "amdxdna ctrlcode ELF patch offset overflows");
          break;
        }
        abs_offset = (uint32_t)total;
      }
      if ((size_t)abs_offset + 8 > ctrl_size || (abs_offset & 3u) != 0) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "amdxdna ctrlcode ELF patch offset %u is outside the %zu-byte "
            "instruction buffer",
            abs_offset, ctrl_size);
        break;
      }
      status = iree_hal_amdxdna_append_patch(host_allocator, &patches,
                                             &patch_count, &patch_capacity,
                                             abs_offset, arg_idx, add_end_addr);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, patches);
    return status;
  }
  *out_patches = patches;
  *out_patch_count = patch_count;
  *out_skipped = skipped;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_build_dpu_slices(
    iree_allocator_t host_allocator, const uint32_t* col_bases,
    uint32_t col_count, size_t ctrl_size,
    iree_hal_amdxdna_ctrlcode_dpu_slice_t** out_slices,
    iree_host_size_t* out_count) {
  *out_slices = NULL;
  *out_count = 0;
  if (col_count == 0 || ctrl_size == 0) return iree_ok_status();
  if (ctrl_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna ctrlcode instruction buffer is too large");
  }
  iree_hal_amdxdna_ctrlcode_dpu_slice_t* slices = NULL;
  iree_host_size_t count = 0;
  size_t covered = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t col = 0; col < col_count && iree_status_is_ok(status); ++col) {
    const uint32_t begin = col_bases[col];
    const uint32_t end = (col + 1 < col_count) ? col_bases[col + 1]
                                               : (uint32_t)ctrl_size;
    if (end < begin) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "amdxdna ctrlcode column %u offset went "
                                "backwards",
                                col);
      break;
    }
    if (end == begin) continue;
    if (count >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna ctrlcode ELF has more than %u occupied columns; START_DPU "
          "indirect packets stop at HSA_MAX_LEVEL1_INDIRECT_ENTRIES",
          IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES);
      break;
    }
    if (count == 0) {
      status = iree_allocator_malloc(
          host_allocator,
          IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES * sizeof(*slices),
          (void**)&slices);
      if (!iree_status_is_ok(status)) break;
    }
    slices[count].uc_index = (uint16_t)col;
    slices[count].reserved = 0;
    slices[count].byte_offset = begin;
    slices[count].byte_size = end - begin;
    covered += (size_t)(end - begin);
    ++count;
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, slices);
    return status;
  }
  if (covered != ctrl_size) {
    iree_allocator_free(host_allocator, slices);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "amdxdna ctrlcode column slices cover %zu of %zu "
                            "instruction bytes",
                            covered, ctrl_size);
  }
  if (count == 1 && slices[0].uc_index != 0) {
    // KMD fill_direct_pkt ignores uc_index. One occupied column is a direct
    // packet, so a hole at column 0 cannot be named. More than one occupied
    // column uses fill_indirect_pkt, which does honor uc_index.
    const uint16_t uc = slices[0].uc_index;
    iree_allocator_free(host_allocator, slices);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna ctrlcode ELF occupies only column %u; a START_DPU direct "
        "packet cannot name a non-zero microcontroller",
        uc);
  }
  if (count > 1) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (slices[i].uc_index >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES) {
        iree_allocator_free(host_allocator, slices);
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "amdxdna ctrlcode column %u is outside the %u microcontrollers "
            "START_DPU indirect packets can name",
            slices[i].uc_index, IREE_HAL_AMDXDNA_CTRLCODE_MAX_DPU_SLICES);
      }
    }
  }
  *out_slices = slices;
  *out_count = count;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_ctrlcode_elf_parse(
    iree_const_byte_span_t elf_bytes, iree_allocator_t host_allocator,
    iree_hal_amdxdna_ctrlcode_elf_t* out_elf) {
  IREE_ASSERT_ARGUMENT(out_elf);
  memset(out_elf, 0, sizeof(*out_elf));
  if (!elf_bytes.data || elf_bytes.data_length < IREE_HAL_AMDXDNA_ELF32_EHDR_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF is missing or truncated");
  }
  const uint8_t* ident = elf_bytes.data;
  if (ident[0] != 0x7F || ident[1] != 'E' || ident[2] != 'L' || ident[3] != 'F') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode image is not an ELF object");
  }
  if (ident[IREE_HAL_AMDXDNA_ELF_EI_CLASS] != IREE_HAL_AMDXDNA_ELF32_CLASS ||
      ident[IREE_HAL_AMDXDNA_ELF_EI_DATA] != IREE_HAL_AMDXDNA_ELF32_DATA_LE ||
      ident[IREE_HAL_AMDXDNA_ELF_EI_VERSION] != IREE_HAL_AMDXDNA_ELF32_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF must be ELF32 little-endian");
  }
  const uint8_t os_abi = ident[IREE_HAL_AMDXDNA_ELF_EI_OSABI];
  const uint8_t abi_version = ident[IREE_HAL_AMDXDNA_ELF_EI_ABIVERSION];
  // OSABI 64 is the original paged AIE2PS layout (`.ctrltext.<col>.<page>`).
  // OSABI 70 (aie2ps_group) uses the same pages plus a trailing `.uc`.
  // OSABI 75 (aiebu AIE4) is the same paged layout; VTD npu3 latency nop.elf
  // is this ABI with `.ctrltext.{0,2,4}.0`.
  const bool paged =
      os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS ||
      os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG ||
      os_abi == IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4;
  if (os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2PS &&
      os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P &&
      os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE2P_CONFIG &&
      os_abi != IREE_HAL_AMDXDNA_CTRLCODE_ELF_OSABI_AIE4) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "amdxdna ctrlcode ELF OSABI %u is not supported",
                            os_abi);
  }

  uint32_t shoff =
      iree_hal_amdxdna_read_u32(elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHOFF);
  uint16_t shentsize =
      iree_hal_amdxdna_read_u16(elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHENTSIZE);
  uint16_t shnum =
      iree_hal_amdxdna_read_u16(elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHNUM);
  uint16_t shstrndx =
      iree_hal_amdxdna_read_u16(elf_bytes.data, IREE_HAL_AMDXDNA_ELF32_E_SHSTRNDX);
  if (shentsize < IREE_HAL_AMDXDNA_ELF32_SHDR_SIZE || shnum == 0 ||
      shnum > IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS || shstrndx >= shnum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna ctrlcode ELF section table is invalid");
  }

  iree_hal_amdxdna_elf_section_t sections[IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS];
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_load_sections(elf_bytes, sections, shnum,
                                                     shoff, shentsize));
  const iree_hal_amdxdna_elf_section_t* shstr = &sections[shstrndx];

  iree_hal_amdxdna_ctrl_page_t pages[IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS];
  iree_host_size_t page_count = 0;
  iree_hal_amdxdna_pad_span_t pads[IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS];
  iree_host_size_t pad_count = 0;
  const iree_hal_amdxdna_elf_section_t* dynsym = NULL;
  const iree_hal_amdxdna_elf_section_t* dynstr = NULL;
  const iree_hal_amdxdna_elf_section_t* xrt_configuration = NULL;
  iree_host_size_t aie2p_ctrltext_count = 0;
  const iree_hal_amdxdna_elf_section_t* aie2p_ctrltext = NULL;

  for (uint16_t i = 0; i < shnum; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_elf_section_name(elf_bytes, shstr, &sections[i], &name));
    if (iree_string_view_equal(name, IREE_SV(".dynsym"))) dynsym = &sections[i];
    if (iree_string_view_equal(name, IREE_SV(".dynstr"))) dynstr = &sections[i];
    if (iree_hal_amdxdna_name_is_xrt_configuration(name)) {
      xrt_configuration = &sections[i];
    }
    if (paged) {
      if (iree_hal_amdxdna_name_is_ctrltext(name) ||
          iree_hal_amdxdna_name_is_ctrldata(name)) {
        uint32_t col = 0;
        uint32_t page = 0;
        IREE_RETURN_IF_ERROR(
            iree_hal_amdxdna_parse_col_page(name, &col, &page));
        iree_hal_amdxdna_ctrl_page_t* slot = iree_hal_amdxdna_find_or_add_page(
            pages, &page_count, col, page);
        if (!slot) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "amdxdna ctrlcode ELF has too many paged "
                                  "sections");
        }
        if (iree_hal_amdxdna_name_is_ctrltext(name)) {
          slot->ctrltext_offset = sections[i].file_offset;
          slot->ctrltext_size = sections[i].size;
          slot->has_ctrltext = true;
        } else {
          slot->ctrldata_offset = sections[i].file_offset;
          slot->ctrldata_size = sections[i].size;
          slot->has_ctrldata = true;
        }
      } else if (iree_hal_amdxdna_name_is_pad(name)) {
        if (pad_count >= IREE_HAL_AMDXDNA_CTRLCODE_MAX_SECTIONS) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "amdxdna ctrlcode ELF has too many pad "
                                  "sections");
        }
        uint32_t col = 0;
        IREE_RETURN_IF_ERROR(iree_hal_amdxdna_parse_pad_col(name, &col));
        pads[pad_count].col = col;
        pads[pad_count].file_offset = sections[i].file_offset;
        pads[pad_count].size = sections[i].size;
        ++pad_count;
      }
    } else if (iree_hal_amdxdna_name_is_ctrltext(name)) {
      ++aie2p_ctrltext_count;
      aie2p_ctrltext = &sections[i];
    }
  }

  uint8_t* ctrl_bytes = NULL;
  size_t ctrl_size = 0;
  uint32_t col_bases[IREE_HAL_AMDXDNA_CTRLCODE_MAX_COLUMNS] = {0};
  uint32_t col_count = 0;
  iree_status_t status = iree_ok_status();
  if (paged) {
    qsort(pages, page_count, sizeof(pages[0]), iree_hal_amdxdna_ctrl_page_cmp);
    status = iree_hal_amdxdna_build_aie2ps_ctrl(
        elf_bytes, host_allocator, pages, page_count, pads, pad_count,
        &ctrl_bytes, &ctrl_size, col_bases, &col_count);
  } else {
    if (aie2p_ctrltext_count != 1 || !aie2p_ctrltext) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna ctrlcode ELF OSABI %u requires exactly one .ctrltext "
          "section; found %" PRIhsz,
          os_abi, aie2p_ctrltext_count);
    }
    if (aie2p_ctrltext->size % sizeof(uint32_t) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "amdxdna ctrlcode ELF .ctrltext size %u is not a "
                              "multiple of 4",
                              aie2p_ctrltext->size);
    }
    size_t capacity = 0;
    status = iree_hal_amdxdna_copy_section_bytes(
        elf_bytes, aie2p_ctrltext->file_offset, aie2p_ctrltext->size,
        host_allocator, &ctrl_bytes, &ctrl_size, &capacity);
    col_count = 1;
  }
  if (!iree_status_is_ok(status)) return status;

  uint32_t* patches = NULL;
  size_t patch_count = 0;
  iree_host_size_t skipped = 0;
  status = iree_hal_amdxdna_parse_relocs(
      elf_bytes, abi_version, paged, sections, shnum, shstr, dynsym, dynstr,
      col_bases, col_count, ctrl_size, host_allocator, &patches, &patch_count,
      &skipped);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, ctrl_bytes);
    return status;
  }

  iree_hal_amdxdna_ctrlcode_dpu_slice_t* dpu_slices = NULL;
  iree_host_size_t dpu_slice_count = 0;
  if (paged) {
    status = iree_hal_amdxdna_build_dpu_slices(host_allocator, col_bases,
                                               col_count, ctrl_size, &dpu_slices,
                                               &dpu_slice_count);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, ctrl_bytes);
      iree_allocator_free(host_allocator, patches);
      return status;
    }
  }

  out_elf->os_abi = os_abi;
  out_elf->ctrl_words = (uint32_t*)ctrl_bytes;
  out_elf->ctrl_word_count = ctrl_size / sizeof(uint32_t);
  out_elf->patches = patches;
  out_elf->patch_count = patch_count;
  out_elf->skipped_non_host_relocs = skipped;
  out_elf->dpu_slices = dpu_slices;
  out_elf->dpu_slice_count = dpu_slice_count;
  out_elf->partition_cols =
      iree_hal_amdxdna_parse_partition_cols(elf_bytes, xrt_configuration);
  return iree_ok_status();
}
