/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_TENSOR_H
#define NPU_DEVICE_CORE_NPU_TENSOR_H

#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/framework/tensor.h"

namespace npu {
struct NpuTensor {
  TF_DISALLOW_COPY_AND_ASSIGN(NpuTensor);

  TFE_TensorHandle* handle;

  explicit NpuTensor(const tensorflow::Tensor& tensor);

  ~NpuTensor();

  static int64_t Dim(void* data, int dim_index, TF_Status* status);

  static int NumDims(void* data, TF_Status* status);

  static void Deallocator(void* data);

  static TFE_CustomDeviceTensorHandleMethods handle_methods;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_TENSOR_H
