/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "om_executor.h"

namespace tensorflow {
namespace {
class LoadAndExecuteOmOp : public OpKernel {
 public:
  explicit LoadAndExecuteOmOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("executor_type", &executor_type_));
  }
  ~LoadAndExecuteOmOp() override = default;

  void Compute(OpKernelContext *ctx) override {
    std::unique_lock<std::mutex> lk(mu_);
    auto input_num = ctx->num_inputs();
    OP_REQUIRES(ctx, input_num > 0,
                errors::Internal("input num should more than 0"));
    model_data_ = ctx->input(input_num - 1).scalar<tstring>()();
    OP_REQUIRES_OK(ctx, Initialize());
    std::vector<Tensor> inputs;
    inputs.reserve(input_num - 1);
    for (int32_t i = 0; i < input_num - 1; i++) {
      inputs.push_back(ctx->input(i));
    }
    std::vector<Tensor> outputs;
    OP_REQUIRES_OK(ctx, executor_->Execute(inputs, outputs));
    OP_REQUIRES(ctx, outputs.size() == static_cast<size_t>(ctx->num_outputs()),
                errors::Internal("Om outputs num mismatch expect ", ctx->num_outputs(), " vs. ", outputs.size()));

    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
      ctx->set_output(i, std::move(outputs[i]));
    }
  }

 private:
  Status Initialize() {
    if (initialized_) {
      return Status::OK();
    }
    // todo: 将om_path_转换为绝对路径
    TF_RETURN_IF_ERROR(OmExecutor::Create(model_data_, executor_));
    initialized_ = true;
    return Status::OK();
  }

  std::mutex mu_;
  bool initialized_{false};
  std::string model_data_;
  std::string executor_type_;  // Reserved

  std::unique_ptr<OmExecutor> executor_;
};
}  // namespace
REGISTER_KERNEL_BUILDER(Name("LoadAndExecuteOm").Device(DEVICE_CPU), LoadAndExecuteOmOp);
}  // namespace tensorflow
