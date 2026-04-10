/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/optimizers/control_flow_conversion_pass.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
namespace {
const string kLowerUsingSwitchMergeAttr = "_lower_using_switch_merge";
}  // namespace

Status ControlFlowConversionPass::Run(const GraphOptimizationPassOptions &options) {
  if (options.partition_graphs != nullptr) {
    return errors::Internal("Lowering If/While ops should happen before partitioning.");
  }
  if (options.graph == nullptr || options.session_options == nullptr) {
    return Status::OK();
  }

  Graph *graph = options.graph->get();
  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(options);
  std::string job = pass_options["job"];
  if (job == "ps" || job == "default") {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : ControlFlowConversionPass.";
    return Status::OK();
  }

  FunctionLibraryDefinition *flib_def = options.flib_def;
  if (flib_def == nullptr) {
    return errors::Internal("Lowering If op requires a FunctionLibraryDefinition to be available.");
  }

  bool use_off_line = pass_options["use_off_line"] == "1";
  bool lower_functional_ops = pass_options["lower_functional_ops"] == "1";
  if (!use_off_line || lower_functional_ops) {
    ADP_LOG(INFO) << "Skip the optimizer";
    return Status::OK();
  }

  // Delete _lower_using_switch_merge before LowerFunctionalOpsPass
  for (int i = 2; i < graph->num_node_ids(); ++i) {
    Node *n = graph->FindNodeId(i);
    if (n == nullptr) {
      continue;
    }
    if (n->IsIfNode() || n->type_string() == "Case" || n->IsWhileNode()) {
      n->ClearAttr(kLowerUsingSwitchMergeAttr);
    }
  }

  std::vector<string> function_names = flib_def->ListFunctionNames();
  for (const std::string &func_name : function_names) {
    const FunctionDef *fdef = flib_def->Find(func_name);
    if (fdef == nullptr) {
      continue;
    }
    for (NodeDef ndef : fdef->node_def()) {
      if (ndef.op() == "If" || ndef.op() == "Case" || ndef.op() == "While") {
        (void) ndef.mutable_attr()->erase(kLowerUsingSwitchMergeAttr);
      }
    }
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, -1, ControlFlowConversionPass);
}  // namespace tensorflow
