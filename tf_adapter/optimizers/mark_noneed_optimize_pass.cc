/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
class MarkNoNeedOptimizePass : public GraphOptimizationPass {
 public:
  MarkNoNeedOptimizePass() = default;
  ~MarkNoNeedOptimizePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;

 private:
  Status ProcessGraph(const std::unique_ptr<Graph> *graph, const FunctionLibraryDefinition *func_lib,
                      const OptimizationPassRegistry::Grouping pass_group_value) const;
};

Status MarkNoNeedOptimizePass::Run(const GraphOptimizationPassOptions &options) {
  if ((options.graph == nullptr && options.partition_graphs == nullptr) || options.flib_def == nullptr) {
    return Status::OK();
  }

  Status s = Status::OK();
  if (options.graph != nullptr) {
    std::unique_ptr<Graph> *graph = options.graph;
    FunctionLibraryDefinition *func_lib = options.flib_def;
    s = ProcessGraph(graph, func_lib, OptimizationPassRegistry::POST_REWRITE_FOR_EXEC);
    if (s != Status::OK()) {
      return s;
    }
  } else if (options.partition_graphs != nullptr) {
    for (auto &pg : *options.partition_graphs) {
      std::unique_ptr<Graph> *graph = &pg.second;
      FunctionLibraryDefinition *func_lib = options.flib_def;
      s = ProcessGraph(graph, func_lib, OptimizationPassRegistry::POST_PARTITIONING);
      if (s != Status::OK()) {
        return s;
      }
    }
  }

  return Status::OK();
}

Status MarkNoNeedOptimizePass::ProcessGraph(const std::unique_ptr<Graph> *graph,
                                            const FunctionLibraryDefinition *func_lib,
                                            const OptimizationPassRegistry::Grouping pass_group_value) const {
  if (graph == nullptr) {
    return Status::OK();
  }

  for (Node *n : graph->get()->nodes()) {
    if (n != nullptr && n->attrs().Find("_NoNeedOptimize")) {
      ADP_LOG(INFO) << "Found mark of noneed optimize on node [" << n->name() << "], skip MarkNoNeedOptimizePass.";
      return Status::OK();
    }
  }

  std::string job;
  std::map<std::string, std::string> pass_options;
  pass_options = NpuAttrs::GetDefaultPassOptions();
  for (Node *n : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->attrs().Find("_NpuOptimizer")) {
      pass_options = NpuAttrs::GetPassOptions(n->attrs());
      break;
    }
  }

  job = pass_options["job"];
  if (job == "ps" || job == "default") {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : MarkNoNeedOptimizePass.";
    return Status::OK();
  }
  if (job == "localhost" && pass_group_value != OptimizationPassRegistry::POST_REWRITE_FOR_EXEC) {
    return Status::OK();
  }
  if (job != "localhost" && pass_group_value != OptimizationPassRegistry::POST_PARTITIONING) {
    return Status::OK();
  }

  bool mix_compile_mode = pass_options["mix_compile_mode"] == "1";
  int iterations_per_loop = std::atoi(pass_options["iterations_per_loop"].c_str());
  ADP_LOG(INFO) << "mix_compile_mode is " << (mix_compile_mode ? "True" : "False");
  ADP_LOG(INFO) << "iterations_per_loop is " << iterations_per_loop;

  for (const auto &func_name : func_lib->ListFunctionNames()) {
    auto *fdef = const_cast<FunctionDef *>(func_lib->Find(func_name));
    if (fdef == nullptr) {
      continue;
    }
    ADP_LOG(INFO) << "Mark function as no need optimize [" << fdef->signature().name() << "]";
    for (NodeDef &ndef : *fdef->mutable_node_def()) {
      (*ndef.mutable_attr())["_NoNeedOptimize"].set_b(true);
    }
  }

  return Status::OK();
}
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 1, MarkNoNeedOptimizePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 100, MarkNoNeedOptimizePass);
}  // namespace tensorflow
