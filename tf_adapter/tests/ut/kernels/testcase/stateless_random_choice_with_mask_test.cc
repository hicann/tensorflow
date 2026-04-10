/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/kernels/aicore/npu_aicore_ops.cc"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include <memory>

namespace tensorflow {
namespace {

PartialTensorShape TShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef &op_def, int in_index, const NodeDef &node_def, NodeDefBuilder *builder) {
    char c = 'a' + (in_index % 26);
    string in_node = string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

TEST(StatelessRandomChoiceWithMaskOpTest, TestStatelessRandomChoiceWithMask) {
  std::vector<DataType> in_types_vec = {DT_BOOL, DT_INT32, DT_INT64, DT_INT64};
  DataTypeSlice input_types(in_types_vec);
  MemoryTypeSlice input_memory_types;
  std::vector<DataType> out_types_vec = {DT_INT32, DT_BOOL};
  DataTypeSlice output_types(out_types_vec);
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context =
      new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types, input_memory_types,
                               output_types, output_memory_types, 1, nullptr);
  StatelessRandomChoiceWithMaskOp stateless_random_choice_with_mask(context);
  OpKernelContext *ctx = nullptr;
  stateless_random_choice_with_mask.Compute(ctx);
  ASSERT_TRUE(stateless_random_choice_with_mask.IsExpensive());
  delete device;
  delete node_def;
  delete op_def;
  delete context;
}

TEST(StatelessRandomChoiceWithMaskOpTest, TestStatelessRandomChoiceWithMaskShapeInference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("StatelessRandomChoiceWithMask", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Input(FakeInputStub(DT_BOOL))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT64))
                  .Input(FakeInputStub(DT_INT64))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({2, 2}), TShape({1}), TShape({1}), TShape({1})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[2,2]", c.DebugString(c.output(0)));
  ASSERT_EQ("[2]", c.DebugString(c.output(1)));
}
}  // namespace
}  // namespace tensorflow
