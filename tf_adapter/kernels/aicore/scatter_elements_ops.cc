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

namespace tensorflow {
namespace {
class ScatterElementsOp : public OpKernel {
 public:
  explicit ScatterElementsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}
  ~ScatterElementsOp() override = default;
  void Compute(OpKernelContext *ctx) override {
    (void) (ctx);
    LOG(INFO) << "in ScatterElements";
  }
  bool IsExpensive() override {
    LOG(INFO) << "in ScatterElements IsExpensive";
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("ScatterElements").Device(DEVICE_CPU), ScatterElementsOp);
}  // namespace
}  // namespace tensorflow
