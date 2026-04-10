/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/common/adapter_logger.h"

namespace tensorflow {
namespace {
class GetNextOp : public OpKernel {
public:
  explicit GetNextOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    ADP_LOG(INFO) << "GetNextOp built " << channel_name_;
  }
  ~GetNextOp() override {
    ADP_LOG(INFO) << "GetNextOp has been destructed";
  }
  void Compute(OpKernelContext *ctx) override {
    (void) ctx;
    ADP_LOG(INFO) << "GetNextOp running";
  }

private:
  std::string channel_name_;
};

REGISTER_KERNEL_BUILDER(Name("GetNext").Device(DEVICE_CPU), GetNextOp);
}  // namespace
}  // namespace tensorflow
