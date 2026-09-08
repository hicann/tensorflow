/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mmpa/mmpa_api.h"
#include "nlohmann/json.hpp"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/optimizers/control_flow_conversion_pass.h"
#include "tf_adapter/optimizers/dp_tf_to_ge_conversion_pass.h"
#include "tf_adapter/optimizers/om_partition_subgraphs_pass.h"
#include "tf_adapter/util/infershape_util.h"
#define private public
#include "tf_adapter/util/generate_report.h"
#include "tf_adapter/util/npu_ops_identifier.h"
#undef private

namespace tensorflow {
#define LOG_COVERAGE_MARK_NO_NEED_PASS MarkNoNeedOptimizePass
class LOG_COVERAGE_MARK_NO_NEED_PASS : public GraphOptimizationPass {
 public:
  MarkNoNeedOptimizePass() = default;
  ~MarkNoNeedOptimizePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;

 private:
  Status ProcessGraph(const std::unique_ptr<Graph> *graph, const FunctionLibraryDefinition *func_lib,
                      const OptimizationPassRegistry::Grouping pass_group_value) const;
};
#undef LOG_COVERAGE_MARK_NO_NEED_PASS

class MarkStartNodePass : public GraphOptimizationPass {
 public:
  MarkStartNodePass() = default;
  ~MarkStartNodePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
  Status TraverseNode(const Node *start_node);
};

namespace {
class LogRemediationCoverageTest : public testing::Test {
 protected:
  void LoadGraph(const string &path) {
    char trusted_path[MMPA_MAX_PATH] = {"\0"};
    ASSERT_EQ(mmRealPath(path.c_str(), trusted_path, MMPA_MAX_PATH), EN_OK);
    GraphDef graph_def;
    ASSERT_TRUE(ReadTextProto(Env::Default(), trusted_path, &graph_def).ok());
    GraphConstructorOptions options;
    graph_ = absl::make_unique<Graph>(OpRegistry::Global());
    ASSERT_TRUE(ConvertGraphDefToGraph(options, graph_def, graph_.get()).ok());
  }

  string GraphEdges(const Graph &graph) const {
    std::vector<string> edges;
    for (const Edge *edge : graph.edges()) {
      if (!edge->src()->IsOp() || !edge->dst()->IsOp()) {
        continue;
      }
      std::ostringstream description;
      description << edge->src()->name() << ":" << edge->src_output() << "->" << edge->dst()->name() << ":"
                  << edge->dst_input();
      edges.emplace_back(description.str());
    }
    return absl::StrJoin(edges, ";");
  }

  GraphOptimizationPassOptions PassOptions(SessionOptions *session_options = nullptr) {
    pass_options_.graph = &graph_;
    pass_options_.session_options = session_options;
    function_library_ = absl::make_unique<FunctionLibraryDefinition>(graph_->flib_def());
    pass_options_.flib_def = function_library_.get();
    return pass_options_;
  }

  string RunOmPass(bool inspect_functions = false) {
    GraphOptimizationPassOptions options = PassOptions();
    EXPECT_TRUE(OMPartitionSubgraphsPass().Run(options).ok());
    if (!inspect_functions) {
      return GraphEdges(*graph_);
    }
    GraphDef graph_def;
    graph_->ToGraphDef(&graph_def);
    string result;
    for (const FunctionDef &function : graph_def.library().function()) {
      Graph function_graph(OpRegistry::Global());
      FunctionDefLibrary library;
      FunctionLibraryDefinition definitions(function_graph.op_registry(), library);
      EXPECT_TRUE(InferShapeUtil::GetSubGraphFromFunctionDef(definitions, function, &function_graph).ok());
      result += GraphEdges(function_graph) + "|";
    }
    return result;
  }

  std::unique_ptr<Graph> graph_;
  std::unique_ptr<FunctionLibraryDefinition> function_library_;
  GraphOptimizationPassOptions pass_options_;
};

TEST_F(LogRemediationCoverageTest, SkipPassesForGraphMarkedNoNeedOptimize) {
  LoadGraph("tf_adapter/tests/ut/optimizers/pbtxt/add_input_pass_no_need_optimize_test.pbtxt");
  const string original = GraphEdges(*graph_);

  SessionOptions session_options;
  GraphOptimizationPassOptions options = PassOptions(&session_options);
  EXPECT_TRUE(MarkNoNeedOptimizePass().Run(options).ok());
  EXPECT_TRUE(ControlFlowConversionPass().Run(options).ok());
  EXPECT_EQ(GraphEdges(*graph_), original);

  EXPECT_TRUE(MarkStartNodePass().Run(options).ok());
  EXPECT_EQ(GraphEdges(*graph_), original);

  EXPECT_TRUE(DpTfToGEConversionPass().Run(options).ok());
  EXPECT_EQ(GraphEdges(*graph_), original);

  EXPECT_TRUE(OMPartitionSubgraphsPass().Run(options).ok());
  EXPECT_EQ(GraphEdges(*graph_), original);
}

TEST_F(LogRemediationCoverageTest, RejectMalformedOpsConfiguration) {
  const std::string path = "/tmp/tf_adapter_invalid_npu_ops.json";
  std::ofstream(path) << "{";
  nlohmann::json ops;
  NpuOpsIdentifier identifier(false, ops);
  EXPECT_EQ(identifier.ParseOps(path, ops), 0);
  EXPECT_EQ(std::remove(path.c_str()), 0);
}

TEST_F(LogRemediationCoverageTest, IgnoreUnsupportedAccumulateInput) {
  LoadGraph("tf_adapter/tests/ut/optimizers/pbtxt/om_test_accumulate.pbtxt");
  Node *accumulate = nullptr;
  Node *unsupported = nullptr;
  for (Node *node : graph_->op_nodes()) {
    if (node->name() == "AccumulateNV2/Internal/_4") {
      accumulate = node;
    } else if (node->name() == "random_uniform") {
      unsupported = node;
    }
  }
  ASSERT_NE(accumulate, nullptr);
  ASSERT_NE(unsupported, nullptr);
  graph_->AddControlEdge(unsupported, accumulate);
  EXPECT_NE(RunOmPass(true).find("AccumulateNV2"), std::string::npos);
}

TEST_F(LogRemediationCoverageTest, HandleStringInputWithoutMaximumSize) {
  LoadGraph("tf_adapter/tests/ut/optimizers/pbtxt/om_test_string_input.pbtxt");
  for (Node *node : graph_->op_nodes()) {
    node->ClearAttr("_op_max_size");
    if (node->attrs().Find("_NpuOptimizer") != nullptr) {
      node->ClearAttr("_iterations_per_loop");
      node->AddAttr("_iterations_per_loop", "2");
    }
  }
  EXPECT_NO_FATAL_FAILURE((void)RunOmPass());
}

TEST_F(LogRemediationCoverageTest, RemovePreviousGenerateReportResult) {
  const std::string path = "check_result.tf.json";
  std::ofstream(path) << "stale";
  GenerateReport report;
  EXPECT_FALSE(std::ifstream(path).good());
}
}  // namespace
}  // namespace tensorflow
