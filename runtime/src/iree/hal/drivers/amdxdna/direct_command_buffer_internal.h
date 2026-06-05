// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "iree/base/api.h"

namespace iree::hal::amdxdna::internal {

// AIE-side aperture base added to every shim-DMA buffer address. Validated for
// npu4 / AIE2P_STRIX_B0 (the only target the chained path is enabled for);
// other AIE generations may use a different offset / BD address layout.
static constexpr uint64_t kDdrAieAddrOffset = 0x80000000ULL;

// Producer RTP lowering may emit WRITE32 values tagged with this sentinel; the
// HAL replaces the low bits with the corresponding dispatch constant before
// handing the transaction to firmware.
static constexpr uint32_t kWrite32ConstantSentinel = 0xA1EC0000u;
static constexpr uint32_t kWrite32ConstantMask = 0xFFFF0000u;

inline uint32_t LoadUnalignedU32(const uint8_t* p) {
  uint32_t value = 0;
  memcpy(&value, p, sizeof(value));
  return value;
}

inline void StoreUnalignedU32(uint8_t* p, uint32_t value) {
  memcpy(p, &value, sizeof(value));
}

// Size in bytes of one XAie transaction operation starting at byte offset `p`.
// Returns 0 on malformed/truncated input.
inline uint32_t TxnOpSize(const uint8_t* b, size_t total, size_t p) {
  if (p >= total) return 0;
  uint8_t op = b[p];
  if (op == 0) {  // WRITE32.
    if (p + 24 > total) return 0;
    return LoadUnalignedU32(b + p + 20);
  }
  if (op == 1) {  // BLOCKWRITE.
    if (p + 16 > total) return 0;
    return LoadUnalignedU32(b + p + 12);
  }
  if (op == 3 || op == 4) {
    if (p + 28 > total) return 0;
    return LoadUnalignedU32(b + p + 24);
  }
  if (op >= 128) {  // Custom op.
    if (p + 8 > total) return 0;
    return LoadUnalignedU32(b + p + 4);
  }
  return 4;
}

inline iree_status_t PatchWrite32Constants(uint32_t* txn, size_t txn_words,
                                           iree_const_byte_span_t constants) {
  if (txn_words < 4) return iree_ok_status();
  uint8_t* b = reinterpret_cast<uint8_t*>(txn);
  size_t total = txn_words * sizeof(uint32_t);
  uint32_t num_ops = txn[2];  // TXN header word 2 = NumOps.
  size_t p = 16;              // Past the 16-byte XAie_TxnHeader.
  for (uint32_t i = 0; i < num_ops; ++i) {
    uint32_t sz = TxnOpSize(b, total, p);
    if (sz == 0 || p + sz > total) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna write32 RTP patch saw malformed transaction op %u at byte "
          "offset %zu",
          i, p);
    }
    if (b[p] == 0) {  // WRITE32: patch sentinel values from HAL constants.
      uint32_t value = LoadUnalignedU32(b + p + 16);
      if ((value & kWrite32ConstantMask) == kWrite32ConstantSentinel) {
        uint32_t constant_index = value & ~kWrite32ConstantMask;
        iree_host_size_t byte_offset =
            static_cast<iree_host_size_t>(constant_index) * sizeof(uint32_t);
        if (byte_offset + sizeof(uint32_t) > constants.data_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "amdxdna write32 RTP constant index %u out of bounds for "
              "%zu-byte constants block",
              constant_index, constants.data_length);
        }
        memcpy(&value, constants.data + byte_offset, sizeof(uint32_t));
        StoreUnalignedU32(b + p + 16, value);
      }
    }
    p += sz;
  }
  return iree_ok_status();
}

// Apply the producer-emitted host patch table to a copy of the control code.
// `patches` is a producer-emitted flat list of (offset, arg_idx, arg_plus)
// triples. For each triple this writes the 48-bit shim-DMA address
// `args[arg_idx] + arg_plus +
// aperture` into the buffer-descriptor address words at byte `offset`: word
// bd[1] (low 32) and the low 16 bits of bd[2] (high). The HAL does NOT parse
// the transaction stream; all XAie-format knowledge stays in the producer; the
// only hardware fact here is the BD address split (a DMA-address ABI).
//
// Returns false on any malformed/out-of-bounds table entry (producer-generated,
// so this is a hard error rather than a recoverable condition).
inline bool ApplyPatchTable(uint32_t* ctrl_code, size_t ctrl_words,
                            const std::vector<uint32_t>& patches,
                            const uint64_t* args, size_t arg_count) {
  if (patches.size() % 3 != 0) return false;
  uint8_t* b = reinterpret_cast<uint8_t*>(ctrl_code);
  size_t total = ctrl_words * sizeof(uint32_t);
  for (size_t i = 0; i < patches.size(); i += 3) {
    uint32_t offset = patches[i];        // byte offset of the BD base word
    uint32_t arg_idx = patches[i + 1];   // index into `args`
    uint32_t arg_plus = patches[i + 2];  // byte addend into that buffer
    if (arg_idx >= arg_count) return false;
    // We touch bd[1] at offset+4 and bd[2] at offset+8 (4 bytes each).
    if (static_cast<size_t>(offset) + 12 > total || (offset & 0x3u) != 0) {
      return false;
    }
    uint32_t* bd = reinterpret_cast<uint32_t*>(b + offset);
    uint64_t base = (static_cast<uint64_t>(bd[2] & 0xFFFF) << 32) | bd[1];
    base += args[arg_idx] + arg_plus + kDdrAieAddrOffset;
    bd[1] = static_cast<uint32_t>(base & 0xFFFFFFFC);
    bd[2] = (bd[2] & 0xFFFF0000) | static_cast<uint32_t>(base >> 32);
  }
  return true;
}

}  // namespace iree::hal::amdxdna::internal

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_INTERNAL_H_
