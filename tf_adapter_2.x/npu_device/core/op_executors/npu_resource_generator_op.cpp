/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_device.h"
#include "npu_managed_buffer.h"
#include "npu_resource_generator_op.h"

namespace npu {
using Format = ge::Format;
NpuResourceGeneratorOp::NpuResourceGeneratorOp(const tensorflow::OpRegistrationData *op_spec,
                                               const tensorflow::NodeDef &ndef, TensorShapes input_shapes)
    : OpExecutor(op_spec, ndef, input_shapes) {
  AssembleInputDesc(input_shapes_, input_dtypes_, attached_attrs_);
}

std::string NpuResourceGeneratorOp::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuResourceGeneratorOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                     int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  if ((!device->SupportedResourceGenerator(Op())) || (!InputTypes().empty()) || (num_outputs != 1)) {
    device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
    return;
  }

  outputs[0] = device->NewDeviceResourceHandle(context, kScalarShape, status);
  NPU_REQUIRES_TFE_OK(status);

  npu::ScopeTensorHandleDeleter scope_handle_deleter;
  TFE_TensorHandle *cpu_output = nullptr;
  device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, &cpu_output, status);
  NPU_REQUIRES_TFE_OK(status);
  scope_handle_deleter.Guard(cpu_output);

  const tensorflow::Tensor *cpu_tensor = nullptr;
  NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(cpu_output, &cpu_tensor));
  const tensorflow::Tensor *npu_tensor = nullptr;
  NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(outputs[0], &npu_tensor));
  auto resource = cpu_tensor->flat<tensorflow::ResourceHandle>()(0);
  const_cast<tensorflow::Tensor *>(npu_tensor)->flat<tensorflow::ResourceHandle>()(0) = resource;

  auto ndef = std::make_shared<tensorflow::NodeDef>(NodeDef());

  tensorflow::SetAttrValue(resource.container(), &ndef->mutable_attr()->at("container"));
  ndef->set_name(WrapResourceName(resource.name()));
  tensorflow::SetAttrValue(ndef->name(), &ndef->mutable_attr()->at("shared_name"));

  device->RecordResourceGeneratorDef(resource, std::make_shared<ResourceGenerator>(ndef, 0));
  DLOG() << "Create resource " << Op() << " " << resource.DebugString() << " by " << ndef->DebugString() << " on NPU";
}
}  // namespace npu
