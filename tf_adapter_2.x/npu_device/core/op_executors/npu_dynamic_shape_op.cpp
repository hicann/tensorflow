/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_dynamic_shape_op.h"

#include "npu_device.h"

namespace npu {
NpuDynamicShapeOp::NpuDynamicShapeOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                                     TensorShapes input_shapes, TensorPartialShapes output_shapes)
    : OpExecutor(op_spec, ndef, input_shapes), output_shapes_(std::move(output_shapes)) {
  AssembleInputDesc(input_shapes_, input_dtypes_, attached_attrs_);
  AssembleOutputDesc(output_shapes_, output_dtypes_, attached_attrs_);
}

std::string NpuDynamicShapeOp::AttachedDebugString() const {
  std::stringstream ss;
  for (size_t i = 0; i < output_dtypes_.size(); i++) {
    ss << "output " << i << " " << tensorflow::DataTypeString(output_dtypes_[i]) << " "
       << output_shapes_[i].DebugString() << std::endl;
  }
  return ss.str();
}

void NpuDynamicShapeOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
}
}  // namespace npu
