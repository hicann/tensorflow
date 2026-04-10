/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/graph/algorithm.h"

#include "op_executors/npu_kernel_registry.h"
#include "npu_global.h"
#include "npu_utils.h"

namespace npu {
static const auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                              TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                              TF_Status *status) {
  (void)ndef;
  (void)num_outputs;
  (void)num_inputs;
  (void)outputs;
  (void)context;
  TFE_TensorHandle *input = inputs[0];
  const tensorflow::Tensor *tensor;
  NPU_CTX_REQUIRES_OK(status, GetTensorHandleTensor(input, &tensor));
  auto handle = tensor->scalar<tensorflow::ResourceHandle>()();
  if (dev->MirroredIterator(handle)) {
    DLOG() << "Start erase iterator provider: " << handle.DebugString();
    dev->EraseIteratorProvider(context, handle);
  }
};

NPU_REGISTER_FALLBACK_HOOK("DeleteIterator", kernel);
}
