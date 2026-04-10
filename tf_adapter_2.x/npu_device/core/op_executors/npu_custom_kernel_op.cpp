/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_custom_kernel_op.h"

#include "npu_device.h"

namespace npu {
NpuCustomKernelOp::NpuCustomKernelOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                                     TensorShapes input_shapes, const NpuCustomKernelFunc &custom_kernel)
    : OpExecutor(op_spec, ndef, input_shapes) {
  cache_strategy_ = CacheStrategy::BY_OP_NAME;
  custom_kernel_ = custom_kernel;
}

std::string NpuCustomKernelOp::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuCustomKernelOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  NPU_CTX_REQUIRES(status, custom_kernel_ != nullptr,
                   tensorflow::errors::Internal(Op(), " custom kernel func is nullptr"));
  custom_kernel_(context, device, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
}
}  // namespace npu
