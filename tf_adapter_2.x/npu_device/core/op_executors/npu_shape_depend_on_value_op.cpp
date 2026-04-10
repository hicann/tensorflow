/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_shape_depend_on_value_op.h"

#include "npu_device.h"
#include "npu_static_shape_op.h"

namespace npu {
std::string NpuShapeDependOnValueOp::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuShapeDependOnValueOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs,
                                      TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                                      TF_Status *status) const {
  TensorPartialShapes partial_shapes;
  auto s = device->InferShape(context, *OpRegistrationData(), NodeDef(), num_inputs, inputs, partial_shapes);
  if (!s.ok()) {
    DLOG() << Op() << " fallback cpu as infer shape failed " << s.ToString();
    device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
    return;
  }

  TensorShapes output_shapes(partial_shapes.size());
  for (size_t i = 0; i < partial_shapes.size(); i++) {
    DLOG() << Op() << " infer shape output " << i << partial_shapes[i].DebugString();
    if (!partial_shapes[i].AsTensorShape(&output_shapes[i])) {
      DLOG() << Op() << " fallback cpu as output " << i << " unknown shape " << partial_shapes[i].DebugString();
      device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
      return;
    }
  }

  NpuStaticShapeOp::RunWithShape(context, device, this, output_shapes, num_inputs, inputs, num_outputs, outputs,
                                 status);
}
}  // namespace npu
