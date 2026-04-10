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
#include "tf_adapter/kernels/npu_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class NpuOnnxGraphOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(NpuOnnxGraphOpTest, TestNpuOnnxGraphOp) {
    std::vector<DataType> in_types_vec = {DT_FLOAT};
    DataTypeSlice input_types(in_types_vec);
    MemoryTypeSlice input_memory_types;
    std::vector<DataType> out_types_vec = {DT_FLOAT};
    DataTypeSlice output_types(out_types_vec);
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    NpuOnnxGraphOp npu_onnx_graph_conv(context);
    OpKernelContext *ctx = nullptr;
    npu_onnx_graph_conv.Compute(ctx);
    ASSERT_FALSE(npu_onnx_graph_conv.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}
