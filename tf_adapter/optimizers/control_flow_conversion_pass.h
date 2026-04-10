/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_CONTROL_FLOW_CONVERSION_PASS_H_
#define TENSORFLOW_CONTROL_FLOW_CONVERSION_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
// Remove _lower_using_switch_merge flag in control flow ops
class ControlFlowConversionPass : public GraphOptimizationPass {
 public:
  ControlFlowConversionPass() = default;
  ~ControlFlowConversionPass() override = default;

  Status Run(const GraphOptimizationPassOptions &options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTROL_FLOW_CONVERSION_PASS_H_
