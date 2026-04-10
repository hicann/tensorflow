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
class LRUCacheV2Op : public OpKernel {
public:
  explicit LRUCacheV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~LRUCacheV2Op() override = default;
  void Compute(OpKernelContext *context) override {
    (void)context;
    ADP_LOG(INFO) << "LRUCacheV2Op Compute running";
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("LRUCacheV2").Device(DEVICE_CPU), LRUCacheV2Op);
} // namespace tensorflow
