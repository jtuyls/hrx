// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

iree_hal_amdxdna_native_buffer_t* FakeBuffer(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_buffer_t*>(value);
}

iree_hal_amdxdna_native_queue_t* FakeQueue(uintptr_t value) {
  return reinterpret_cast<iree_hal_amdxdna_native_queue_t*>(value);
}

iree_hal_amdxdna_chain_cmd MakeCmd(
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t device_addr,
    size_t repeat_count = 1) {
  iree_hal_amdxdna_chain_cmd cmd;
  cmd.ctrl_words = {0xA, 0xB, 0xC};
  cmd.binding_buffers = {buffer};
  cmd.binding_device_addrs = {device_addr};
  cmd.binding_offsets = {4};
  cmd.binding_lengths = {128};
  cmd.src_asm_inst = reinterpret_cast<const std::vector<uint32_t>*>(0x1000);
  cmd.src_patches = reinterpret_cast<const std::vector<uint32_t>*>(0x2000);
  cmd.src_cu_idx.index = 7;
  cmd.src_constants = {1, 2, 3, 4};
  cmd.src_use_native_partial_elf = true;
  cmd.repeat_count = repeat_count;
  return cmd;
}

iree_hal_amdxdna_chain_group MakeGroup(
    std::vector<iree_hal_amdxdna_chain_cmd>&& cmds) {
  iree_hal_amdxdna_chain_group group;
  group.queue = FakeQueue(0x3000);
  group.native_partial_elf = true;
  group.cmds = std::move(cmds);
  return group;
}

iree_hal_amdxdna_chain_group MakeGroup1(iree_hal_amdxdna_chain_cmd cmd) {
  std::vector<iree_hal_amdxdna_chain_cmd> cmds;
  cmds.push_back(std::move(cmd));
  return MakeGroup(std::move(cmds));
}

iree_hal_amdxdna_chain_command_cache_entry MakeCacheEntry(
    iree_hal_amdxdna_chain_group group, uint32_t max_slots = 24) {
  iree_hal_amdxdna_chain_command_cache_entry entry;
  entry.group = std::move(group);
  entry.max_slots = max_slots;
  entry.chains.emplace_back();
  return entry;
}

TEST(ChainCommandCacheTest, DeviceMatchRejectsSameAddressDifferentBuffer) {
  auto cached = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x80000000);
  auto entry = MakeCacheEntry(MakeGroup1(std::move(cached)));
  auto group = MakeGroup1(std::move(fresh));

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_device_matches(
      entry, group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, DeviceMatchAcceptsSameAddressAndBuffer) {
  auto cached = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto entry = MakeCacheEntry(MakeGroup1(std::move(cached)));
  auto group = MakeGroup1(std::move(fresh));

  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_device_matches(
      entry, group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, ShapeMatchAllowsDifferentBufferForRebind) {
  auto cached = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000);
  auto fresh = MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x80000000);
  auto entry = MakeCacheEntry(MakeGroup1(std::move(cached)));
  auto group = MakeGroup1(std::move(fresh));

  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_shape_matches(
      entry, group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, DescriptorMatchExpandsRepeatCounts) {
  auto compact = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000,
                         /*repeat_count=*/3);
  auto entry = MakeCacheEntry(MakeGroup1(std::move(compact)));
  std::vector<iree_hal_amdxdna_chain_cmd> expanded;
  expanded.push_back(MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000));
  expanded.push_back(MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000));
  expanded.push_back(MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000));
  auto group = MakeGroup(std::move(expanded));

  EXPECT_TRUE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      entry, group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, DescriptorMatchRejectsChangedBindings) {
  auto compact = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000,
                         /*repeat_count=*/3);
  auto entry = MakeCacheEntry(MakeGroup1(std::move(compact)));
  std::vector<iree_hal_amdxdna_chain_cmd> expanded;
  expanded.push_back(MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000));
  expanded.push_back(MakeCmd(FakeBuffer(0x20), /*device_addr=*/0x80000000));
  expanded.push_back(MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000));
  auto group = MakeGroup(std::move(expanded));

  EXPECT_FALSE(iree_hal_amdxdna_chain_command_cache_descriptor_matches(
      entry, group, /*max_slots=*/24));
}

TEST(ChainCommandCacheTest, LogicalCommandCountSaturatesOnOverflow) {
  auto first = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000,
                       std::numeric_limits<size_t>::max());
  auto second = MakeCmd(FakeBuffer(0x10), /*device_addr=*/0x80000000, 2);
  std::vector<iree_hal_amdxdna_chain_cmd> cmds;
  cmds.push_back(std::move(first));
  cmds.push_back(std::move(second));
  auto group = MakeGroup(std::move(cmds));

  EXPECT_EQ(iree_hal_amdxdna_chain_group_logical_command_count(group),
            std::numeric_limits<size_t>::max());
}

}  // namespace
