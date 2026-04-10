/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/util/host_allocator.h"
#include "gtest/gtest.h"

namespace tensorflow {
class HostAllocatorTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(HostAllocatorTest, allocate_test)  {
  int64_t a = 10;
  HostAllocator host_allocat(static_cast<void *>(&a));
  std::string name = host_allocat.Name();
  EXPECT_EQ(name, "host_allocator");
  void *ptr = host_allocat.AllocateRaw(0, 0);
}
} //end tensorflow
