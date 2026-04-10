/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <deque>
#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
static const int64 kMicrosToMillis = 1000;
static std::atomic<int> graph_run_num(1);

std::set<string> StringSplit(const string &str, const string &pattern) {
  std::set<string> resultSet;
  string::size_type pos2 = str.find(pattern);
  string::size_type pos1 = 0;
  while (pos2 != string::npos) {
    (void) resultSet.insert(str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + pattern.size();
    pos2 = str.find(pattern, pos1);
  }
  if (pos1 != str.length()) {
    (void) resultSet.insert(str.substr(pos1));
  }
  return resultSet;
}

class MarkStartNodePass : public GraphOptimizationPass {
 public:
  MarkStartNodePass() = default;
  ~MarkStartNodePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
  Status TraverseNode(const Node *start_node);
};

Status MarkStartNodePass::Run(const GraphOptimizationPassOptions &options) {
  int graph_num = graph_run_num++;

  bool not_need_process =
    (options.graph == nullptr || options.flib_def == nullptr || options.session_options == nullptr);
  if (not_need_process) {
    return Status::OK();
  }

  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(options);
  std::string job = pass_options["job"];
  bool skip_flag = (job == "ps" || job == "default" || job == "localhost") ;
  if (skip_flag) {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : MarkStartNodePass.";
    return Status::OK();
  }

  std::unique_ptr<Graph> *graph = options.graph;

  for (Node *n : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->attrs().Find("_NoNeedOptimize")) {
      ADP_LOG(INFO) << "Found mark of noneed optimize on node [" << n->name() << "], skip MarkStartNodePass.";
      return Status::OK();
    }

    if (n->attrs().Find("_StartNodeName")) {
      ADP_LOG(INFO) << "Found mark of startnode optimize on node [" << n->name() << "], skip MarkStartNodePass.";
      return Status::OK();
    }
  }

  int64 startTime = InferShapeUtil::GetCurrentTimestap();

  if (kDumpGraph) {
    GraphDef ori_graph_def;
    graph->get()->ToGraphDef(&ori_graph_def);
    string ori_model_path = GetDumpPath() + "BeforeMarkStartNodeAttr_";
    string omodel_path = ori_model_path + std::to_string(graph_num) + ".pbtxt";
    (void)WriteTextProto(Env::Default(), omodel_path, ori_graph_def);
  }

  for (Node *start_node : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(start_node);
    std::string src_device_name = start_node->assigned_device_name();
    if (!src_device_name.empty() && src_device_name.find("/job:ps") == std::string::npos) {
      for (Node *n : start_node->out_nodes()) {
        std::string device_name = n->assigned_device_name();
        if (device_name.find("/job:ps") != std::string::npos) {
          std::string start_node_name;
          std::set<string> start_nodes_name;
          auto start_node_attr_value = start_node->attrs().Find("_StartNodeName");
          if (start_node_attr_value != nullptr) {
            start_nodes_name = StringSplit(start_node_attr_value->s(), ";");
          }
          (void) start_nodes_name.insert(start_node->name());
          for (const auto &name : start_nodes_name) {
            start_node_name += name;
            start_node_name += ";";
          }
          start_node->AddAttr("_StartNodeName", start_node_name);

          auto n_attr_value = n->attrs().Find("_StartNodeName");
          if (n_attr_value != nullptr) {
            std::set<string> nodes_name = StringSplit(n_attr_value->s(), ";");
            for (const auto &name : nodes_name) {
              (void) start_nodes_name.insert(name);
            }
          }
          for (const auto &name : start_nodes_name) {
            start_node_name += name;
            start_node_name += ";";
          }
          n->AddAttr("_StartNodeName", start_node_name);
          Status s = TraverseNode(n);
          if (s != Status::OK()) {
            return s;
          }
        }
      }
    }
  }

  if (kDumpGraph) {
    GraphDef omg_graph_def;
    graph->get()->ToGraphDef(&omg_graph_def);
    string tmpmodel_path = GetDumpPath() + "AfterMarkStartNodeAttr_";
    string tmodel_path = tmpmodel_path + std::to_string(graph_num) + ".pbtxt";
    (void)WriteTextProto(Env::Default(), tmodel_path, omg_graph_def);
  }
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "MarkStartNodePass_" << std::to_string(graph_num) << " success. ["
                << ((endTime - startTime) / kMicrosToMillis) << " ms]";

  return Status::OK();
}

Status MarkStartNodePass::TraverseNode(const Node *start_node) {
  REQUIRES_NOT_NULL(start_node);
  Status s = Status::OK();
  for (Node *n : start_node->out_nodes()) {
    REQUIRES_NOT_NULL(n);
    std::string start_node_name;
    std::set<string> start_nodes_name;
    auto start_node_attr_value = start_node->attrs().Find("_StartNodeName");
    if (start_node_attr_value != nullptr) {
      start_nodes_name = StringSplit(start_node_attr_value->s(), ";");
    }

    auto n_attr_value = n->attrs().Find("_StartNodeName");
    if (n_attr_value != nullptr) {
      std::set<string> nodes_name = StringSplit(n_attr_value->s(), ";");
      for (const auto &name : nodes_name) {
        (void) start_nodes_name.insert(name);
      }
    }
    for (const auto &name : start_nodes_name) {
      start_node_name += name;
      start_node_name += ";";
    }
    n->AddAttr("_StartNodeName", start_node_name);
    s = TraverseNode(n);
    if (s != Status::OK()) {
      ADP_LOG(INFO) << "traverse node : " << start_node->name() << " can't to add start node name.";
      return s;
    }
  }
  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, -1, MarkStartNodePass);
}  // namespace tensorflow
