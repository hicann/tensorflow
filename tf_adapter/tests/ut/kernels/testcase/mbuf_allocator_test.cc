/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include <stdlib.h>
#include "gtest/gtest.h"

namespace tensorflow {
namespace {
class MbufAllocatorTest : public testing::Test {
  protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
    const int64_t kRuntimeTensorDescSize = 1024UL;
    const size_t alignment = 64;
};

TEST_F(MbufAllocatorTest, EnableMbufAllocatorTest) {
  tensorflow::int64 enable_mbuf_allocator = 0;
  (void)tensorflow::ReadInt64FromEnvVar("ENABLE_MBUF_ALLOCATOR", 0, &enable_mbuf_allocator);
  if (enable_mbuf_allocator == 1) {
    Allocator* a = cpu_allocator();
    EXPECT_EQ(a->Name(), "MbufAllocator");

    void* raw_ptr = a->AllocateRaw(alignment, kRuntimeTensorDescSize);
    auto *alloc_ptr = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(raw_ptr) -
                                                     kRuntimeTensorDescSize);
    for (int i = 0; i < kRuntimeTensorDescSize / sizeof (int32_t); i++) {
      int32_t alloc_ptr_i = alloc_ptr[i];
      ADP_LOG(INFO) << i << " " << alloc_ptr_i; // no dump
    }
    a->DeallocateRaw(raw_ptr);
    unsetenv("ENABLE_MBUF_ALLOCATOR");
  }
}
}
} //end tensorflow
