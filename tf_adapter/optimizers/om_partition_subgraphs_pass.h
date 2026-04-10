/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_OM_PARTITION_SUBGRAPHS_PASS_H_
#define TENSORFLOW_OM_PARTITION_SUBGRAPHS_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace OMSplitter {
Status MarkForPartition(const GraphOptimizationPassOptions &options, int &clusterNum, bool mix_compile_mode,
                        int graph_num, const FunctionLibraryDefinition *func_lib,
                        std::map<std::string, std::string> pass_options,
                        std::map<std::string, std::string> &graph_options);

bool IsNpuSupportingNode(const NodeDef &node_def, bool mix_compile_mode,
                         const FunctionLibraryDefinition *func_lib, bool support_const = false);
bool IsNpuSupportingNode(const Node *node, bool mix_compile_mode, const FunctionLibraryDefinition *func_lib,
                         bool support_const = false);
}  // namespace OMSplitter

class OMPartitionSubgraphsPass : public GraphOptimizationPass {
 public:
  OMPartitionSubgraphsPass() = default;
  ~OMPartitionSubgraphsPass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;

 private:
  Status ProcessGraph(std::unique_ptr<Graph> *graph, FunctionLibraryDefinition *func_lib,
                      const OptimizationPassRegistry::Grouping pass_group_value) const;
  Status AccumulateNFusion(Graph *graph_in, Node *node) const;
  void GetGraphConfig(const Node &node, bool enable_dp, std::map<std::string, std::string> &graph_options) const;
  void ParseInputShapeRange(const std::string dynamic_inputs_shape_range, bool enable_dp,
                            std::map<std::string, std::string> &graph_options) const;
  Status ProcessGetNext(Node &node, const std::string enable_dp, std::vector<Node *> &remove_nodes,
                        Graph &graph_in) const;
  Status SplitUnaryOpsComposition(Graph *graph, Node *node) const;
  Status CopyVarsBetweenGeOp(Graph *graph) const;
  Status CopyConstBetweenGeOp(Graph *graph) const;
  void InheritAttributes(Node &node) const;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_OM_PARTITION_SUBGRAPHS_PASS_H_
