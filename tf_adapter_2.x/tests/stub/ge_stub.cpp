/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_parser.h"
#include "ge/ge_api.h"
#include "ge/ge_api_wrapper.h"

namespace ge {
Graph GraphUtilsEx::CreateGraphFromComputeGraph(const ge::ComputeGraphPtr compute_graph) {
  Graph graph;
  graph.graph = compute_graph->graph;
  return graph;
}

ge::ComputeGraphPtr GraphUtilsEx::GetComputeGraph(const Graph &graph) {
  auto compute_graph = std::make_shared<ge::ComputeGraph>("");
  compute_graph->graph = graph.graph;
  return compute_graph;
}

Status Session::AddGraph(uint32_t graphId, const Graph &graph) {
  graphs_[graphId] = graph.graph;
  graph_need_rebuild_[graphId] = false;
  return SUCCESS;
}

Status Session::AddGraph(uint32_t graphId, const Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options) {
  graphs_[graphId] = graph.graph;
  graph_need_rebuild_[graphId] = false;
  return SUCCESS;
}

Status Session::RemoveGraph(uint32_t graphId) {
  graphs_.erase(graphId);
  graph_need_rebuild_.erase(graphId);
  return SUCCESS;
}

Status Session::RunGraphAsync(uint32_t graphId, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback) {
  // std::function<void(Status, std::vector<ge::Tensor> &)>;
  auto graph = graphs_[graphId];
  std::vector<ge::Tensor> outputs;

  if (graph == nullptr) {
    callback(ge::FAILED, outputs);
    return ge::FAILED;
  }
  for (const auto &node : graph->op_nodes()) {
    if (node->IsRetval()) {
      int index = 0;
      const tensorflow::AttrValue *attr = node->attrs().Find(npu::kInputDesc);
      if (attr == nullptr) {
        const tensorflow::Edge *edge = nullptr;
        node->input_edge(0, &edge);
        if (edge == nullptr || edge->src()->attrs().Find(npu::kOutputDesc) == nullptr) {
          LOG(ERROR) << "Can not mock tensor for " << node->name();
          continue;
        }
        attr = edge->src()->attrs().Find(npu::kOutputDesc);
        index = edge->src_output();
      }

      auto &desc_attr = attr->list().func(index).attr();

      std::vector<int64_t> dims;
      if (desc_attr.find(npu::kShape) != desc_attr.end()) {
        tensorflow::AttrValue shape_value = desc_attr.at(npu::kShape);
        for (int i = 0; i < shape_value.list().i_size(); i++) {
          auto dim_size = shape_value.list().i(i);
          if (dim_size < 0) {
            dim_size = 1;
          }
          dims.push_back(dim_size);
        }
      }
      ge::DataType dtype = ge::DT_FLOAT;
      if (desc_attr.find(npu::kType) != desc_attr.end()) {
        tensorflow::AttrValue shape_value = desc_attr.at(npu::kType);
        dtype = static_cast<ge::DataType>(shape_value.i());
      }

      outputs.emplace_back(ge::Tensor());
      size_t size = tensorflow::DataTypeSize(static_cast<tensorflow::DataType>(dtype));
      for (auto dim : dims) {
        size *= dim;
      }
      std::vector<uint8_t> data(size);
      outputs.back().SetData(data.data(), size);
      outputs.back().SetTensorDesc(ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, dtype));
    }
  }
  callback(ge::SUCCESS, outputs);
  return ge::SUCCESS;
}

bool Session::IsGraphNeedRebuild(uint32_t graphId) { return graph_need_rebuild_[graphId]; }

size_t ComputeGraph::GetAllNodesSize() const { return graph->num_op_nodes(); }
size_t ComputeGraph::GetInputSize() const { return 1U; }
size_t ComputeGraph::GetOutputSize() const { return 1U; }
ge::AscendString GEGetErrorMsgV2() { return ge::AscendString("ERROR"); }
ge::AscendString GEGetWarningMsgV2() { return ge::AscendString("WARNING"); }

Status GEInitialize(const std::map<ge::AscendString, ge::AscendString> &options) { return SUCCESS; }

Status GEFinalize() { return SUCCESS; }

const char *GetFormatName(Format format) {
  static const char *names[FORMAT_END] = {
    "NCHW",
    "NHWC",
    "ND",
    "NC1HWC0",
    "FRACTAL_Z",
    "NC1C0HWPAD",  // 5
    "NHWC1C0",
    "FSR_NCHW",
    "FRACTAL_DECONV",
    "C1HWNC0",
    "FRACTAL_DECONV_TRANSPOSE",  // 10
    "FRACTAL_DECONV_SP_STRIDE_TRANS",
    "NC1HWC0_C04",
    "FRACTAL_Z_C04",
    "CHWN",
    "DECONV_SP_STRIDE8_TRANS",  // 15
    "HWCN",
    "NC1KHKWHWC0",
    "BN_WEIGHT",
    "FILTER_HWCK",
    "LOOKUP_LOOKUPS",  // 20
    "LOOKUP_KEYS",
    "LOOKUP_VALUE",
    "LOOKUP_OUTPUT",
    "LOOKUP_HITS",
    "C1HWNCoC0",  // 25
    "MD",
    "NDHWC",
    "UNKNOWN",  // FORMAT_FRACTAL_ZZ
    "FRACTAL_NZ",
    "NCDHW",  // 30
    "DHWCN",
    "NDC1HWC0",
    "FRACTAL_Z_3D",
    "CN",
    "NC",  // 35
    "DHWNC",
    "FRACTAL_Z_3D_TRANSPOSE",
    "FRACTAL_ZN_LSTM",
    "FRACTAL_Z_G",
    "UNKNOWN",  // 40, FORMAT_RESERVED
    "UNKNOWN",  // FORMAT_ALL
    "UNKNOWN",  // FORMAT_NULL
    "ND_RNN_BIAS",
    "FRACTAL_ZN_RNN",
  };
  if (format >= FORMAT_END) {
    return "UNKNOWN";
  }
  return names[format];
}
}  // namespace ge

ge::Graph GeApiWrapper_CreateGraphFromComputeGraph(const ge::ComputeGraphPtr &compute_graph) {
  return ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}

size_t GeApiWrapper_GetComputeGraphInputSize(const ge::Graph &graph) {
  return ge::GraphUtilsEx::GetComputeGraph(graph)->GetInputSize();
}

size_t GeApiWrapper_GetComputeGraphOutputSize(const ge::Graph &graph) {
  return ge::GraphUtilsEx::GetComputeGraph(graph)->GetOutputSize();
}

ge::ComputeGraphPtr GeApiWrapper_MakeComputeGraphPtr(const char *graph_name) {
  return std::make_shared<ge::ComputeGraph>(graph_name);
}

size_t GeApiWrapper_GetAllNodesSize(const ge::ComputeGraphPtr &graph_ptr) {
  return graph_ptr->GetAllNodesSize();
}
