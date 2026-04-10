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
class AdamApplyOneWithDecayAssignOp : public OpKernel {
 public:
  explicit AdamApplyOneWithDecayAssignOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~AdamApplyOneWithDecayAssignOp() override = default;
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "AdamApplyOneWithDecayAssignOp Compute, num_inputs: " << context->num_inputs();
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("AdamApplyOneWithDecayAssign").Device(DEVICE_CPU), AdamApplyOneWithDecayAssignOp);
}  // namespace tensorflow
