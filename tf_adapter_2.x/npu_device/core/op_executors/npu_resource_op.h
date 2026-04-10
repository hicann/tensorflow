/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_OP_EXECUTORS_NPU_RESOURCE_OP_H
#define NPU_DEVICE_CORE_OP_EXECUTORS_NPU_RESOURCE_OP_H

#include "npu_op_executor.h"
#include "npu_concrete_graph.h"

namespace npu {
class NpuResourceOp : public OpExecutor {
  using HashKey = uint64_t;

 public:
  NpuResourceOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                TensorShapes input_shapes);

  const std::string &Type() const override {
    const static std::string kType = "NpuResourceOp";
    return kType;
  }
  ~NpuResourceOp() = default;

  void RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
               TFE_TensorHandle **outputs, TF_Status *status) const override;
 protected:
  std::string AttachedDebugString() const override;
 private:
  static HashKey Hash(const std::vector<tensorflow::ResourceHandle> &handles);

  void GetFuncSpec(const std::vector<tensorflow::ResourceHandle> &handles,
                   std::shared_ptr<NpuConcreteGraph> *spec) const;

  void CacheFuncSpec(const std::vector<tensorflow::ResourceHandle> &handles,
                     std::shared_ptr<NpuConcreteGraph> spec) const;

  tensorflow::mutex mutable shared_lock;
  std::map<HashKey, std::shared_ptr<NpuConcreteGraph>> mutable handles_to_specs_ TF_GUARDED_BY(shared_lock);
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_OP_EXECUTORS_NPU_RESOURCE_OP_H
