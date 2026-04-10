/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_unwrap.h"

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "npu_micros.h"
#include "npu_tensor.h"

namespace npu {
tensorflow::Status GetTensorHandleShape(TFE_TensorHandle *handle, tensorflow::TensorShape &shape) {
  tensorflow::PartialTensorShape partial_shape;
  TF_RETURN_IF_ERROR(tensorflow::unwrap(handle)->Shape(&partial_shape));
  NPU_REQUIRES(partial_shape.AsTensorShape(&shape),
               tensorflow::errors::InvalidArgument("Shape ", partial_shape.DebugString(), " not fully defined"));
  return tensorflow::Status::OK();
}

tensorflow::Status GetTensorHandleTensor(TFE_TensorHandle *handle, const tensorflow::Tensor **tensor) {
  if (IsNpuTensorHandle(handle)) {
    void *dev_buf = dynamic_cast<tensorflow::CustomDeviceTensorHandle *>(tensorflow::unwrap(handle))->DevicePointer();
    return tensorflow::TensorHandleFromInterface(
             tensorflow::unwrap(reinterpret_cast<npu::NpuTensor *>(dev_buf)->handle))
      ->Tensor(tensor);
  } else {
    return tensorflow::TensorHandleFromInterface(tensorflow::unwrap(handle))->Tensor(tensor);
  }
}
}  // namespace npu
