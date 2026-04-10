/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_SET_VAR_FORMAT_PASS_H_
#define TENSORFLOW_SET_VAR_FORMAT_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class SetVarFormatPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions &options) override;
  Status AssignFormatToVarOutNodes(Node *node) const;
  Status GetFormat(const Node *node, string &format) const;
  Status AssignApplyMomentumInNodesFormat(const Node *node, const string &var_format) const;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_SET_VAR_FORMAT_PASS_H_
