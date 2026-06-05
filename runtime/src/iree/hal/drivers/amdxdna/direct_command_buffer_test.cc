// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

namespace internal = iree::hal::amdxdna::internal;

class DirectCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amdxdna_device_options_initialize(&device_params_);
    device_ =
        new iree_hal_amdxdna_device(&device_params_, iree_allocator_system());
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(), &block_pool_);
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_arena_block_pool_deinitialize(&device_->block_pool);
    delete device_;
  }

  iree_hal_amdxdna_device_params device_params_;
  iree_hal_amdxdna_device* device_ = nullptr;
  iree_arena_block_pool_t block_pool_;
};

TEST_F(DirectCommandBufferTest, CreateRejectsUnretainedMode) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  iree_status_t status = iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer);

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
  EXPECT_EQ(command_buffer, nullptr);
}

TEST_F(DirectCommandBufferTest, CreateAllowsRetainedMode) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer));

  ASSERT_NE(command_buffer, nullptr);
  iree_hal_command_buffer_release(command_buffer);
}

TEST(DirectCommandBufferPatchTest, ApplyPatchTableWritesAieAddresses) {
  std::vector<uint32_t> control_code = {0, 0, 0xCAFE0000u, 0};
  uint64_t args[] = {0x1234};

  ASSERT_TRUE(internal::ApplyPatchTable(control_code.data(),
                                        control_code.size(), {0, 0, 0x10}, args,
                                        IREE_ARRAYSIZE(args)));

  uint64_t expected = internal::kDdrAieAddrOffset + args[0] + 0x10;
  EXPECT_EQ(control_code[1], static_cast<uint32_t>(expected & 0xFFFFFFFCULL));
  EXPECT_EQ(control_code[2],
            0xCAFE0000u | static_cast<uint32_t>(expected >> 32));
}

TEST(DirectCommandBufferPatchTest, ApplyPatchTableWritesMultipleAieAddresses) {
  std::vector<uint32_t> control_code = {
      0, 0, 0xCAFE0000u, 0, 0, 0xBABE0000u,
  };
  uint64_t args[] = {0x1234, 0x100000000ull};

  ASSERT_TRUE(internal::ApplyPatchTable(
      control_code.data(), control_code.size(), {0, 0, 0x10, 12, 1, 0x20}, args,
      IREE_ARRAYSIZE(args)));

  uint64_t expected0 = internal::kDdrAieAddrOffset + args[0] + 0x10;
  uint64_t expected1 = internal::kDdrAieAddrOffset + args[1] + 0x20;
  EXPECT_EQ(control_code[1], static_cast<uint32_t>(expected0 & 0xFFFFFFFCull));
  EXPECT_EQ(control_code[2],
            0xCAFE0000u | static_cast<uint32_t>(expected0 >> 32));
  EXPECT_EQ(control_code[4], static_cast<uint32_t>(expected1 & 0xFFFFFFFCull));
  EXPECT_EQ(control_code[5],
            0xBABE0000u | static_cast<uint32_t>(expected1 >> 32));
}

TEST(DirectCommandBufferPatchTest, ApplyPatchTableRejectsMalformedInputs) {
  std::vector<uint32_t> control_code = {0, 0, 0, 0};
  uint64_t args[] = {0};

  EXPECT_FALSE(internal::ApplyPatchTable(control_code.data(),
                                         control_code.size(), {0, 0}, args,
                                         IREE_ARRAYSIZE(args)));
  EXPECT_FALSE(internal::ApplyPatchTable(control_code.data(),
                                         control_code.size(), {0, 1, 0}, args,
                                         IREE_ARRAYSIZE(args)));
  EXPECT_FALSE(internal::ApplyPatchTable(control_code.data(),
                                         control_code.size(), {1, 0, 0}, args,
                                         IREE_ARRAYSIZE(args)));
  EXPECT_FALSE(internal::ApplyPatchTable(control_code.data(),
                                         control_code.size(), {8, 0, 0}, args,
                                         IREE_ARRAYSIZE(args)));
}

static std::vector<uint32_t> MakeWrite32Txn(uint32_t value) {
  std::vector<uint32_t> txn(10, 0);
  txn[2] = 1;      // XAie_TxnHeader.NumOps.
  txn[8] = value;  // WRITE32 value at byte offset 16 + 16.
  txn[9] = 24;     // WRITE32 op size at byte offset 16 + 20.
  return txn;
}

TEST(DirectCommandBufferPatchTest, TxnOpSizeHandlesKnownAndMalformedOps) {
  std::vector<uint8_t> write32(24, 0);
  write32[0] = 0;
  internal::StoreUnalignedU32(write32.data() + 20, 24);
  EXPECT_EQ(internal::TxnOpSize(write32.data(), write32.size(), 0), 24u);
  EXPECT_EQ(internal::TxnOpSize(write32.data(), write32.size() - 1, 0), 0u);

  std::vector<uint8_t> block_write(16, 0);
  block_write[0] = 1;
  internal::StoreUnalignedU32(block_write.data() + 12, 16);
  EXPECT_EQ(internal::TxnOpSize(block_write.data(), block_write.size(), 0),
            16u);
  EXPECT_EQ(internal::TxnOpSize(block_write.data(), block_write.size() - 1, 0),
            0u);

  std::vector<uint8_t> custom(8, 0);
  custom[0] = 128;
  internal::StoreUnalignedU32(custom.data() + 4, 8);
  EXPECT_EQ(internal::TxnOpSize(custom.data(), custom.size(), 0), 8u);
  EXPECT_EQ(internal::TxnOpSize(custom.data(), custom.size() - 1, 0), 0u);

  std::vector<uint8_t> fixed_size(4, 2);
  EXPECT_EQ(internal::TxnOpSize(fixed_size.data(), fixed_size.size(), 0), 4u);
  EXPECT_EQ(internal::TxnOpSize(fixed_size.data(), fixed_size.size(),
                                fixed_size.size()),
            0u);
}

TEST(DirectCommandBufferPatchTest, PatchWrite32ConstantsReplacesSentinel) {
  std::vector<uint32_t> txn =
      MakeWrite32Txn(internal::kWrite32ConstantSentinel | 1u);
  uint32_t constants[] = {0x11111111u, 0x22222222u};

  IREE_ASSERT_OK(internal::PatchWrite32Constants(
      txn.data(), txn.size(),
      iree_make_const_byte_span(constants, sizeof(constants))));

  EXPECT_EQ(txn[8], constants[1]);
}

TEST(DirectCommandBufferPatchTest, PatchWrite32ConstantsRejectsMalformedTxn) {
  std::vector<uint32_t> txn(5, 0);
  txn[2] = 1;

  iree_status_t status = internal::PatchWrite32Constants(
      txn.data(), txn.size(), iree_const_byte_span_empty());

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST(DirectCommandBufferPatchTest,
     PatchWrite32ConstantsRejectsOutOfBoundsConstant) {
  std::vector<uint32_t> txn =
      MakeWrite32Txn(internal::kWrite32ConstantSentinel | 1u);
  uint32_t constants[] = {0x11111111u};

  iree_status_t status = internal::PatchWrite32Constants(
      txn.data(), txn.size(),
      iree_make_const_byte_span(constants, sizeof(constants)));

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

}  // namespace
