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
#include "tf_adapter/kernels/aicpu/npu_cpu_ops.cc"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

PartialTensorShape TShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
              NodeDefBuilder* builder) {
    char c = 'a' + (in_index % 26);
    string in_node =  string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

TEST(OCROpsTest, TestBatchEnqueue) {
    std::vector<DataType> in_types_vec = {DT_INT32, DT_INT32};
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
    BatchEnqueueOp cache(context);
    ASSERT_TRUE(cache.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestOCRRecognitionPreHandle) {
    std::vector<DataType> in_types_vec = {DT_UINT8, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    DataTypeSlice input_types(in_types_vec);
    MemoryTypeSlice input_memory_types;
    std::vector<DataType> out_types_vec = {DT_UINT8, DT_INT32, DT_INT32};
    DataTypeSlice output_types(out_types_vec);
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    OCRRecognitionPreHandleOp cache(context);
    ASSERT_TRUE(cache.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestOCRDetectionPreHandle) {
    std::vector<DataType> in_types_vec = {DT_UINT8};
    DataTypeSlice input_types(in_types_vec);
    MemoryTypeSlice input_memory_types;
    std::vector<DataType> out_types_vec = {DT_UINT8, DT_FLOAT, DT_FLOAT};
    DataTypeSlice output_types(out_types_vec);
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    OCRDetectionPreHandleOp cache(context);
    ASSERT_TRUE(cache.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestOCRIdentifyPreHandle) {
    std::vector<DataType> in_types_vec = {DT_UINT8, DT_INT32, DT_INT32};
    DataTypeSlice input_types(in_types_vec);
    MemoryTypeSlice input_memory_types;
    std::vector<DataType> out_types_vec = {DT_UINT8};
    DataTypeSlice output_types(out_types_vec);
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    OCRIdentifyPreHandleOp cache(context);
    ASSERT_TRUE(cache.IsExpensive());
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestBatchEnqueueShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BatchEnqueue", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_INT32)
                  .Attr("batch_size", 8)
                  .Attr("queue_name", "TEST")
                  .Attr("pad_mode", "REPLICATE")
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_UINT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({5}), TShape({})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRRecognitionPreHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRRecognitionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("batch_size", 8)
                  .Attr("data_format", "NHWC")
                  .Attr("pad_mode", "REPLICATE")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3}), TShape({3}), TShape({3}), TShape({3}), TShape({3})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3, 3, 3})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInference1) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3, 3, 3})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({10}), TShape({3}), TShape({3, 3})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInference1) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3}), TShape({3}), TShape({3, 3})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestBatchDilatePolysShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BatchDilatePolys", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({1}), TShape({1}), TShape({1}), TShape({1}),TShape({1}),TShape({1}),TShape({1})},{}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRFindContoursShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRFindContours", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("value_mode", 0)
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({2})},{}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestDequeueShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Dequeue", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("queue_name", "TEST")
                  .Attr("output_type", DT_UINT8)
                  .Attr("output_shape", {2})
                  .Input(FakeInputStub(DT_UINT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRDetectionPostHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPostHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({3}), TShape({3}), TShape({3}), TShape({3})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestResizeAndClipPolysInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("ResizeAndClipPolys", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({})}, {}, {}, {});
  ASSERT_TRUE(reg->shape_inference_fn(&c).ok());
}
}  // namespace
}  // namespace tensorflow
