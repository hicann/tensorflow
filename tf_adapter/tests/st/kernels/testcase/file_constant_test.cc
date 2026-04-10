/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include "tf_adapter/kernels/npu_ops.cc"
#include "gtest/gtest.h"
#include <memory>
namespace tensorflow {
namespace {
TEST(FileConstantTest, TestFileConstant) {
  std::vector<DataType> in_types_vec = {DT_FLOAT};
  DataTypeSlice input_types(in_types_vec);
  MemoryTypeSlice input_memory_types;
  std::vector<DataType> out_types_vec = {DT_FLOAT};
  DataTypeSlice output_types(out_types_vec);
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  FileConstant file_constant(context);
  OpKernelContext *ctx = nullptr;
  file_constant.Compute(ctx);
  ASSERT_FALSE(file_constant.IsExpensive());
  delete device;
  delete node_def;
  delete op_def;
  delete context;
}

TEST(FileConstantTest, FileConstantShapeInference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("FileConstant", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("file_path", "test")
                  .Attr("file_id", "test")
                  .Attr("shape", {3,2})
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&def));
  const std::vector<shape_inference::ShapeHandle> input_shapes = {};
  shape_inference::InferenceContext c(0, &def, op_def, input_shapes,
                                      {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,2]", c.DebugString(c.output(0)));
}
} // namespace
} // namespace tensorflow
