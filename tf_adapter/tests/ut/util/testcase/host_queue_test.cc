/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/tensor.h"
#include "tf_adapter/util/npu_plugin.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/host_queue.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace tensorflow {
namespace {
class HostQueueTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(HostQueueTest, HostQueueSendData) {
  std::string name = "host_queue_001";
  uint32_t depth = 128U;
  uint32_t queue_id = 0U;
  TF_CHECK_OK(HostQueueInit(name, depth, queue_id));
  Tensor a(DT_UINT32, TensorShape({2, 2}));
  a.flat<uint32_t>()(0) = 1;
  a.flat<uint32_t>()(1) = 1;
  a.flat<uint32_t>()(2) = 1;
  a.flat<uint32_t>()(3) = 1;
  std::vector<Tensor> tensors{a};
  void *buff = nullptr;
  TF_CHECK_OK(MappingTensor2Buff(ACL_TENSOR_DATA_TENSOR, tensors, buff));
  bool need_resend = false;
  Status s = HostQueueSendData(queue_id, buff, need_resend);
  ASSERT_TRUE(s.ok());
  HostQueueDestroy(queue_id);
}

TEST_F(HostQueueTest, HostQueueEndOfSequence) {
  std::string name = "host_queue_001";
  uint32_t depth = 128U;
  uint32_t queue_id = 0U;
  TF_CHECK_OK(HostQueueInit(name, depth, queue_id));
  void *buff = nullptr;
  TF_CHECK_OK(MappingTensor2Buff(ACL_TENSOR_DATA_TENSOR, {}, buff));
  bool need_resend = false;
  Status s = HostQueueSendData(queue_id, buff, need_resend);
  ASSERT_TRUE(s.ok());
  HostQueueDestroy(queue_id);
}
}
} // end tensorflow
