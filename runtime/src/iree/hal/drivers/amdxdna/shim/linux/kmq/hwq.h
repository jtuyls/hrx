// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _HWQ_XDNA_H_
#define _HWQ_XDNA_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fence.h"
#include "host_queue.h"
#include "hwctx.h"

namespace shim_xdna {
// DRM_IOCTL_AMDXDNA_EXEC_CMD takes a flat arg-BO handle array. Keep the shim
// submit path and HAL preflight checks on the same ceiling.
static constexpr size_t kMaxArgBosPerCommand = 1024;

struct bo;
struct device;
struct pdev;
struct hw_q {
  const hw_ctx* m_hwctx;
  const pdev& m_pdev;
  uint32_t m_queue_boh;
  // Number of DRM_IOCTL_AMDXDNA_EXEC_CMD submissions issued through this queue
  // (incremented by issue_command). Lets tests assert the count of on-device
  // submits, e.g. that a deferred ERT_CMD_CHAIN actually batched the
  // recorded dispatches into one submit rather than fanning out.
  std::atomic<uint64_t> m_exec_cmd_count{0};

  hw_q(const device& device);
  ~hw_q();

  // Returns: >0 the command was signaled (check its ert state for COMPLETED vs
  // a terminal error), 0 on ETIME timeout, or -errno on a hard wait-ioctl
  // failure.
  int wait_command(bo*, uint32_t timeout_ms) const;
  int submit_wait(const fence_handle*, uint64_t* out_state);
  int submit_wait(const std::vector<fence_handle*>&, uint64_t* out_last_state);
  int submit_signal(const fence_handle*, uint64_t* out_state);
  void bind_hwctx(const hw_ctx* ctx);
  void unbind_hwctx();
  // Allocate the AIE4 user-mode host queue BO. CREATE_HWCTX consumes
  // m_queue_boh. Returns 0 or an errno.
  int init_umq();
  // Map the doorbell returned by CREATE_HWCTX. A missing doorbell is not an
  // error: submit then uses DRM_IOCTL_AMDXDNA_EXEC_CMD (kernel-mode).
  int map_doorbell(uint32_t doorbell_offset);
  // Returns 0 on success or the failing errno from the EXEC_CMD ioctl / UMQ
  // submit.
  int issue_command(bo*);
  uint64_t exec_cmd_count() const {
    return m_exec_cmd_count.load(std::memory_order_relaxed);
  }

 private:
  std::unique_ptr<bo> m_umq_bo;
  volatile struct host_queue_header* m_umq_hdr = nullptr;
  volatile struct host_queue_packet* m_umq_pkt = nullptr;
  volatile struct host_indirect_data* m_umq_indirect_buf = nullptr;
  uint64_t m_indirect_paddr = 0;
  volatile uint32_t* m_doorbell = nullptr;
  size_t m_doorbell_map_size = 0;
  uint32_t m_umq_slots = 0;

  int issue_umq_exec_buf(bo* cmd_bo);
  int get_next_umq_slot(uint32_t* out_slot);
};

int poll_command(bo*);

}  // namespace shim_xdna

#endif  // _HWQ_XDNA_H_
