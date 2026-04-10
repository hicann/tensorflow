/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/common/adapter_logger.h"

namespace tensorflow {
class BasicLSTMCellCStateGradOp : public OpKernel {
 public:
  explicit BasicLSTMCellCStateGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~BasicLSTMCellCStateGradOp() override = default;
  void Compute(OpKernelContext* context) override {
    (void) context;
    ADP_LOG(INFO) << "BasicLSTMCellCStateGradOp Compute";
  }
  bool IsExpensive() override { return false; }
};

class BasicLSTMCellWeightGradOp : public OpKernel {
 public:
  explicit BasicLSTMCellWeightGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~BasicLSTMCellWeightGradOp() override = default;
  void Compute(OpKernelContext* context) override {
    (void) context;
    ADP_LOG(INFO) << "BasicLSTMCellWeightGradOp Compute";
  }
  bool IsExpensive() override { return false; }
};

class BasicLSTMCellInputGradOp : public OpKernel {
 public:
  explicit BasicLSTMCellInputGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~BasicLSTMCellInputGradOp() override = default;
  void Compute(OpKernelContext* context) override {
    (void) context;
    ADP_LOG(INFO) << "BasicLSTMCellInputGradOp Compute";
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("BasicLSTMCellCStateGrad").Device(DEVICE_CPU), BasicLSTMCellCStateGradOp);
REGISTER_KERNEL_BUILDER(Name("BasicLSTMCellWeightGrad").Device(DEVICE_CPU), BasicLSTMCellWeightGradOp);
REGISTER_KERNEL_BUILDER(Name("BasicLSTMCellInputGrad").Device(DEVICE_CPU), BasicLSTMCellInputGradOp);
}  // namespace tensorflow
