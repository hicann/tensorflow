/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_OP_EXECUTORS_NPU_UNSUPPORTED_OP_H
#define NPU_DEVICE_CORE_OP_EXECUTORS_NPU_UNSUPPORTED_OP_H

#include "npu_op_executor.h"

namespace npu {
class NpuUnsupportedOp : public OpExecutor {
 public:
  NpuUnsupportedOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                   TensorShapes input_shapes, std::string reason)
      : OpExecutor(op_spec, ndef, input_shapes), fallback_reason_(std::move(reason)) {}

  const std::string &Type() const override {
    const static std::string kType = "NpuUnsupportedOp";
    return kType;
  }
  ~NpuUnsupportedOp() = default;

  void RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
               TFE_TensorHandle **outputs, TF_Status *status) const override;
 protected:
  std::string AttachedDebugString() const override;
 private:
  std::string fallback_reason_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_OP_EXECUTORS_NPU_UNSUPPORTED_OP_H
