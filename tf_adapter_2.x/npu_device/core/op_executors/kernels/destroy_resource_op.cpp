/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "op_executors/npu_kernel_registry.h"
#include "npu_utils.h"

namespace npu {
static const auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                              TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                              TF_Status *status) {
  if (!IsNpuTensorHandle(inputs[0])) {
    dev->FallbackCPU(context, ndef, num_inputs, inputs, num_outputs, outputs, status);
    return;
  }

  const tensorflow::Tensor *npu_tensor = nullptr;
  NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(inputs[0], &npu_tensor));

  tensorflow::Tensor cpu_tensor(npu_tensor->dtype(), npu_tensor->shape());
  for (int j = 0; j < npu_tensor->NumElements(); j++) {
    cpu_tensor.flat<tensorflow::ResourceHandle>()(j) =
      const_cast<tensorflow::Tensor *>(npu_tensor)->flat<tensorflow::ResourceHandle>()(j);
  }

  npu::ScopeTensorHandleDeleter scope_handle_deleter;
  auto handle = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
  scope_handle_deleter.Guard(handle);

  dev->FallbackCPU(context, ndef, num_inputs, &handle, num_outputs, outputs, status);
};

NPU_REGISTER_CUSTOM_KERNEL("DestroyResourceOp", kernel);
}  // namespace npu
