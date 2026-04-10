/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef HISI_OFFLINE

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
class LogTimeStampOP : public OpKernel {
 public:
  explicit LogTimeStampOP(OpKernelConstruction *ctx) : OpKernel(ctx) { ADP_LOG(INFO) << "new LogTimeStampOP"; }
  ~LogTimeStampOP() override = default;
  void Compute(OpKernelContext *ctx) override { ADP_LOG(INFO) << "in LogTimeStampOP"; }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("LogTimeStamp").Device(DEVICE_CPU), LogTimeStampOP);
}  // namespace tensorflow

#endif  // HISI_OFFLINE
