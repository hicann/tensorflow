/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_tensor.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

namespace npu {
NpuTensor::NpuTensor(const tensorflow::Tensor& tensor)
    : handle(tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor))) {}

NpuTensor::~NpuTensor() { TFE_DeleteTensorHandle(handle); }

int64_t NpuTensor::Dim(void* data, int dim_index, TF_Status* status) {
  return TFE_TensorHandleDim(reinterpret_cast<NpuTensor*>(data)->handle, dim_index, status);
}

int NpuTensor::NumDims(void* data, TF_Status* status) {
  return TFE_TensorHandleNumDims(reinterpret_cast<NpuTensor*>(data)->handle, status);
}

void NpuTensor::Deallocator(void* data) { delete reinterpret_cast<NpuTensor*>(data); }

TFE_CustomDeviceTensorHandle NpuTensor::handle_methods = []() {
  TFE_CustomDeviceTensorHandle handle_methods_;
  handle_methods_.num_dims = &NpuTensor::NumDims;
  handle_methods_.dim = &NpuTensor::Dim;
  handle_methods_.deallocator = &NpuTensor::Deallocator;
  return handle_methods_;
}();
}  // namespace npu
