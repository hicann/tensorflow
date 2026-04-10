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
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tf_adapter/kernels/aicpu/npu_embedding_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
namespace {
PartialTensorShape TShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef &op_def, int in_index, const NodeDef &node_def,
              NodeDefBuilder *builder) {
    char c = 'a' + (in_index % 26);
    string in_node = string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

class NpuCpuOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST(EmbeddingOpsTest, InitEmbeddingHashmapV2ShapeInfer) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("InitEmbeddingHashmapV2", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                    .Attr("bucket_size", 10)
                    .Attr("load_factor", 80)
                    .Attr("embedding_dim", 2)
                    .Attr("dtype", DT_FLOAT)
                    .Input(FakeInputStub(DT_INT32))
                    .Finalize(&def));
  shape_inference::InferenceContext c(
        0, &def, op_def,
        {TShape({6})},
        {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(EmbeddingOpsTest, DeinitEmbeddingHashmapV2ShapeInfer) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DeinitEmbeddingHashmapV2", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                    .Input(FakeInputStub(DT_INT32))
                    .Finalize(&def));
  shape_inference::InferenceContext c(
        0, &def, op_def,
        {TShape({6})},
        {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(EmbeddingOpsTest, TableToResourceV2ShapeInfer) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("TableToResourceV2", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                    .Input(FakeInputStub(DT_INT32))
                    .Finalize(&def));
  shape_inference::InferenceContext c(
        0, &def, op_def,
        {TShape({6})},
        {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(EmbeddingOpsTest, EmbeddingHashmapImportShapeInfer) {
    const OpRegistrationData *reg;
    TF_CHECK_OK(OpRegistry::Global()->LookUp("EmbeddingHashmapImport", &reg));
    OpDef op_def = reg->op_def;
    NodeDef def;
    TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                    .Attr("embedding_dim", 4)
                    .Attr("num", 1)
                    .Input(FakeInputStub(DT_STRING))
                    .Input(FakeInputStub(DT_INT32))
                    .Input(FakeInputStub(DT_INT64))
                    .Input(FakeInputStub(DT_STRING))
                    .Input(FakeInputStub(DT_INT64))
                    .Finalize(&def));
    shape_inference::InferenceContext c(
        0, &def, op_def,
        {TShape({}), TShape({}), TShape({6})},
        {}, {}, {});
    ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}
}
}
