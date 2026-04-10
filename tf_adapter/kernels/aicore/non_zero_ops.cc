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
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
template<typename T>
class NonZeroOP : public OpKernel {
 public:
  explicit NonZeroOP(OpKernelConstruction *ctx) : OpKernel(ctx) {
    LOG(INFO) << "new NonZeroOP";
  }
  ~NonZeroOP() override = default;
  void Compute(OpKernelContext *ctx) override {
    (void) ctx;
    LOG(INFO) << "compute in NonZeroOP";
  }
  bool IsExpensive() override {
    LOG(INFO) << "in NonZero IsExpensive";
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("NonZero").Device(DEVICE_CPU), NonZeroOP<float>);
}  // namespace tensorflow
