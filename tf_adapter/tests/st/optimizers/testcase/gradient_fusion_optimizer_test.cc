/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tf_adapter/optimizers/grad_fusion_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {
class GradientFusionOptimizerTest : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(GradientFusionOptimizerTest, RunOptimizer) {
  GrapplerItem item;
  item.graph = GraphDef();
  GraphDef output;
  const Status status = GradFusionOptimizer().Optimize(nullptr, item, &output);
  EXPECT_EQ(status, Status::OK());
}
} // end grappler
} // end tensorflow
