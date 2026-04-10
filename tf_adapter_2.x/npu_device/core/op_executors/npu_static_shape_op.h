/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_OP_EXECUTORS_NPU_STATIC_SHAPE_OP_H
#define NPU_DEVICE_CORE_OP_EXECUTORS_NPU_STATIC_SHAPE_OP_H

#include "npu_op_executor.h"

namespace npu {
class NpuStaticShapeOp : public OpExecutor {
 public:
  NpuStaticShapeOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                   TensorShapes input_shapes, TensorShapes output_shapes);

  const std::string &Type() const override {
    const static std::string kType = "NpuStaticShapeOp";
    return kType;
  }
  ~NpuStaticShapeOp() = default;

  void RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
               TFE_TensorHandle **outputs, TF_Status *status) const override;

  static void RunWithShape(TFE_Context *context, NpuDevice *device, const OpExecutor *spec, TensorShapes output_shapes,
                           int num_inputs, TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                           TF_Status *status);

  TensorShapes OutputShapes() const { return output_shapes_; }
 protected:
  std::string AttachedDebugString() const override;
 private:
  TensorShapes output_shapes_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_OP_EXECUTORS_NPU_STATIC_SHAPE_OP_H
