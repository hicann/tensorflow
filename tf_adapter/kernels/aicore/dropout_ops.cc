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
class DropOutDoMaskOp : public OpKernel {
 public:
  explicit DropOutDoMaskOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DropOutDoMaskOp() override {}
  void Compute(OpKernelContext *context) override {
    (void) context;
    ADP_LOG(INFO) << "DropOutDoMaskOp Compute ";
  }
  bool IsExpensive() override {
    return false;
  }
};

class DropOutGenMaskOp : public OpKernel {
 public:
  explicit DropOutGenMaskOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DropOutGenMaskOp() override {}
  void Compute(OpKernelContext *context) override {
    (void) context;
    ADP_LOG(INFO) << "DropOutGenMaskOp Compute";
  }
  bool IsExpensive() override {
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("DropOutGenMask").Device(DEVICE_CPU), DropOutGenMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutGenMaskV3").Device(DEVICE_CPU), DropOutGenMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutGenMaskV4").Device(DEVICE_CPU), DropOutGenMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutDoMask").Device(DEVICE_CPU), DropOutDoMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutDoMaskV3").Device(DEVICE_CPU), DropOutDoMaskOp);
}  // namespace tensorflow
