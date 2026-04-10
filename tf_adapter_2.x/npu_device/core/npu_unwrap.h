/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_UNWRAP_H
#define NPU_DEVICE_CORE_NPU_UNWRAP_H

#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"

#include "tensorflow/core/common_runtime/eager/custom_device.h"

namespace npu {
template <typename T>
static T *Unwrap(const tensorflow::Tensor *tensor) {
  return reinterpret_cast<T *>(const_cast<char *>(tensor->tensor_data().data()));
}

inline tensorflow::EagerContext *UnwrapCtx(TFE_Context *context) {
  return tensorflow::ContextFromInterface(tensorflow::unwrap(context));
}

inline const tensorflow::AttrBuilder *UnwrapAttrs(const TFE_OpAttrs *attrs) {
  return static_cast<const tensorflow::AttrBuilder *>(tensorflow::unwrap(attrs));
}

inline bool IsNpuTensorHandle(TFE_TensorHandle *handle) {
  return tensorflow::CustomDeviceTensorHandle::classof(tensorflow::unwrap(handle));
}

inline bool IsCpuTensorHandle(TFE_TensorHandle *handle) {
  return !tensorflow::CustomDeviceTensorHandle::classof(tensorflow::unwrap(handle));
}

tensorflow::Status GetTensorHandleShape(TFE_TensorHandle *handle, tensorflow::TensorShape &shape);

tensorflow::Status GetTensorHandleTensor(TFE_TensorHandle *handle, const tensorflow::Tensor **tensor);
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_UNWRAP_H
