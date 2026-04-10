/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/public/version.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/node_def_builder.h"

#include "ascendcl_stub.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace {
class DummyDevice : public DeviceBase {
 public:
  DummyDevice() : DeviceBase(Env::Default()) {}
  bool RequiresRecordingAccessedTensors() const override {
    return false;
  }
  Allocator *GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
  }
};

class LoadAndExecuteOmTest : public testing::Test {
 public:
  void SetUp() override {
    device.reset(new DummyDevice());
    lib_def.reset(new FunctionLibraryDefinition(OpRegistry::Global(), {}));
    pflr.reset(new ProcessFunctionLibraryRuntime(nullptr, Env::Default(), TF_GRAPH_DEF_VERSION, lib_def.get(), {},
                                                 nullptr, nullptr));
    params.device = device.get();
    params.function_library = pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  }

  Status CreateKernel(const NodeDef &def) {
    Status status;
    kernel = CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(), def, TF_GRAPH_DEF_VERSION, &status);
    params.op_kernel = kernel.get();
    return status;
  }

  Status Run(std::vector<Tensor> &tensors) {
    gtl::InlinedVector<TensorValue, 4> inputs;
    for (auto &tensor : tensors) {
      inputs.push_back(TensorValue(&tensor));
    }
    params.inputs = &inputs;
    auto ctx = absl::make_unique<OpKernelContext>(&params, 1);
    kernel->Compute(ctx.get());
    return ctx->status();
  }

  std::vector<Tensor> CreateInputTensors() {
    std::vector<Tensor> tensors;
    TensorShape tf_shape;
    tf_shape.AddDim(2);
    tf_shape.AddDim(1);
    Tensor tensor = Tensor(DT_FLOAT, tf_shape);
    tensors.emplace_back(tensor);
    tensors.emplace_back(tensor);
    Tensor tensor_var = Tensor(DT_STRING, {});
    tensors.emplace_back(tensor_var);
    return tensors;
  }

  Status CreateAndRunOmKernel() {
    NodeDef def;
    NodeDefBuilder::NodeOut input1("input1", 0, DT_FLOAT);
    NodeDefBuilder::NodeOut input2("input1", 1, DT_FLOAT);
    NodeDefBuilder::NodeOut var_input("model_data", 2, DT_STRING);
    auto status = tensorflow::NodeDefBuilder("om", "LoadAndExecuteOm")
                          .Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>{input1, input2})
                          .Input(var_input)
                          .Attr("Tin", DataTypeVector{DT_FLOAT, DT_FLOAT})
                          .Attr("output_dtypes", DataTypeVector{DT_FLOAT})
                          .Finalize(&def);
    if (!status.ok()) {
      return status;
    }
    auto kernel_status = CreateKernel(def);
    if (!kernel_status.ok()) {
      return kernel_status;
    }
    auto tensors = CreateInputTensors();
    return Run(tensors);
  }

  void TearDown() override {}

  std::unique_ptr<OpKernel> kernel;
  std::unique_ptr<DeviceBase> device;
  OpKernelContext::Params params;
  std::unique_ptr<FunctionLibraryDefinition> lib_def;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr;
};

/**********************************************
REGISTER_OP("LoadAndExecuteOm")
    .Input("inputs: Tin")
    .Attr("Tin: list(type) >= 0")
    .Output("outputs: output_dtypes")
    .Attr("output_dtypes: list(type) >= 0")
    .Attr("om_path: string")
    .Attr("executor_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);
**********************************************/

TEST_F(LoadAndExecuteOmTest, TestOmNodeExecuteSuccess) {
  ASSERT_EQ(CreateAndRunOmKernel(), Status::OK());
}

TEST_F(LoadAndExecuteOmTest, TestOmNodeExecuteDynamicBatchSuccess) {
  SetDynamicType(0);
  ASSERT_EQ(CreateAndRunOmKernel(), Status::OK());
  SetDynamicType(-1);
}

TEST_F(LoadAndExecuteOmTest, TestOmNodeExecuteDynamicOutputSuccess) {
  SetOutputDynamic(true);
  ASSERT_EQ(CreateAndRunOmKernel(), Status::OK());
  SetOutputDynamic(false);
}

TEST_F(LoadAndExecuteOmTest, TestOmNodeExecuteDynamicOutputZero) {
  SetOutputDynamic(true);
  SetOutputNeedNull(true);
  RegAclRunGraphStub(nullptr);
  ASSERT_EQ(CreateAndRunOmKernel(), Status::OK());
  SetOutputDynamic(false);
  SetOutputNeedNull(false);
}
}  // namespace
}  // namespace tensorflow
