// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host-queue packet layout used by AIE4 CERT (UMQ). Byte-compatible with
// amd/xdna-driver src/shim/umq/host_queue.h.

#ifndef IREE_HAL_DRIVERS_AMDXDNA_SHIM_LINUX_KMQ_HOST_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_SHIM_LINUX_KMQ_HOST_QUEUE_H_

#include <stdint.h>

#define HSA_MAX_LEVEL1_INDIRECT_ENTRIES (6)

#define LAST_CMD (0)
#define NOT_LAST_CMD (1)

enum host_queue_packet_opcode {
  HOST_QUEUE_PACKET_EXEC_BUF = 1,
  HOST_QUEUE_PACKET_TEST = 2,
  HOST_QUEUE_PACKET_EXIT = 3,
};

struct exec_buf {
  uint32_t dtrace_buf_host_addr_low;
  uint32_t dpu_control_code_host_addr_low;
  uint32_t dpu_control_code_host_addr_high;
  uint16_t args_len;
  uint16_t dtrace_buf_host_addr_high;
  uint32_t args_host_addr_low;
  uint32_t args_host_addr_high;
};

#define HOST_QUEUE_MAJOR_VERSION 1
#define HOST_QUEUE_MINOR_VERSION 0

struct host_queue_header {
  uint64_t read_index;
  struct {
    uint16_t major;
    uint16_t minor;
  } version;
  uint32_t capacity;
  uint64_t padding0[6];
  uint64_t write_index;
  uint64_t padding1[6];
  uint64_t data_address;
};

enum host_queue_packet_type {
  HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  HOST_QUEUE_PACKET_TYPE_INVALID = 1,
};

struct common_header {
  uint8_t type;
  uint8_t reserved;
  uint8_t opcode;
  uint8_t chain_flag;
  uint16_t count;
  uint8_t distribute;
  uint8_t indirect;
};

struct xrt_packet_header {
  struct common_header common_header;
  uint64_t completion_signal;
};

struct host_indirect_packet_entry {
  uint32_t host_addr_low;
  uint32_t host_addr_high : 25;
  uint32_t uc_index : 7;
};

struct host_queue_packet {
  struct xrt_packet_header xrt_header;
  uint32_t data[12];
};

struct host_indirect_data {
  struct common_header header;
  struct exec_buf payload;
};

// KMD payload after the mandatory CU mask for ERT_START_DPU
// (drivers/accel/amdxdna/amdxdna_ctx.h::amdxdna_cmd_start_dpu). Same layout as
// XRT ert_dpu_data (dtrace, instruction VA, size, uc_index, chained).
struct amdxdna_cmd_start_dpu {
  uint64_t dtrace_buffer;
  uint64_t instruction_buffer;
  uint32_t instruction_buffer_size;
  uint16_t uc_index;
  uint16_t chained;
};

#endif  // IREE_HAL_DRIVERS_AMDXDNA_SHIM_LINUX_KMQ_HOST_QUEUE_H_
