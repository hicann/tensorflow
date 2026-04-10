/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "tf_adapter/kernels/aicore/scatter_elements_ops.cc"
#include "tf_adapter/kernels/aicpu/npu_cpu_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class ScatterElementsOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

class ScatterElementsV2OpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(ScatterElementsOpTest, TestScatterElements) {
    std::vector<DataType> in_types_vec = {DT_INT32, DT_INT32, DT_INT32};
    DataTypeSlice input_types(in_types_vec);
    MemoryTypeSlice input_memory_types;
    std::vector<DataType> out_types_vec = {DT_INT32};
    DataTypeSlice output_types(out_types_vec);
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    ScatterElementsOp scatter_elements(context);
    OpKernelContext *ctx = nullptr;
    scatter_elements.Compute(ctx);
    ASSERT_FALSE(scatter_elements.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
TEST_F(ScatterElementsV2OpTest, TestScatterElementsV2) {
    std::vector<DataType> in_types_vec = {DT_INT32, DT_INT32, DT_INT32};
    DataTypeSlice input_types(in_types_vec);
    MemoryTypeSlice input_memory_types;
    std::vector<DataType> out_types_vec = {DT_INT32};
    DataTypeSlice output_types(out_types_vec);
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    ScatterElementsV2Op scatter_elements_V2(context);
    OpKernelContext *ctx = nullptr;
    scatter_elements_V2.Compute(ctx);
    ASSERT_FALSE(scatter_elements_V2.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}
