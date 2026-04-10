/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_stub.h"
#include "callback_executor.h"
#include "ge/ge_api_wrapper.h"
#include "ge/ge_ir_build.h"

extern "C" const char *aclGetCustomOpLibPath() {
  return ge::GetCustomPathStub();
}

namespace ge {
std::string g_custon_path;
void SetCustomPathStub(std::string path) {
  g_custon_path = path;
}
const char* GetCustomPathStub() {
  return g_custon_path.c_str();
}
namespace {
const std::map<uint32_t, ge::DataType> data_type_map = {
  {tensorflow::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
  {tensorflow::DataType::DT_HALF, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_INT8, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_INT16, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_UINT16, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_UINT8, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_INT32, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_INT64, ge::DataType::DT_INT64},
  {tensorflow::DataType::DT_UINT32, ge::DataType::DT_UINT32},
  {tensorflow::DataType::DT_UINT64, ge::DataType::DT_UINT64},
  {tensorflow::DataType::DT_BOOL, ge::DataType::DT_BOOL},
  {tensorflow::DataType::DT_DOUBLE, ge::DataType::DT_DOUBLE},
  {tensorflow::DataType::DT_COMPLEX64, ge::DataType::DT_COMPLEX64},
  {tensorflow::DataType::DT_QINT8, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_QUINT8, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_QINT32, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_QINT16, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_QUINT16, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_COMPLEX128, ge::DataType::DT_COMPLEX128},
  {tensorflow::DataType::DT_RESOURCE, ge::DataType::DT_RESOURCE},
  {tensorflow::DataType::DT_BFLOAT16, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_STRING, ge::DataType::DT_STRING},
  {tensorflow::DataType::DT_FLOAT_REF, ge::DataType::DT_FLOAT},
  {tensorflow::DataType::DT_DOUBLE_REF, ge::DataType::DT_DOUBLE},
  {tensorflow::DataType::DT_INT32_REF, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_INT8_REF, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_UINT8_REF, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_INT16_REF, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_UINT16_REF, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_COMPLEX64_REF, ge::DataType::DT_COMPLEX64},
  {tensorflow::DataType::DT_QINT8_REF, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_QUINT8_REF, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_QINT32_REF, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_QINT16_REF, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_QUINT16_REF, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_COMPLEX128_REF, ge::DataType::DT_COMPLEX128},
  {tensorflow::DataType::DT_RESOURCE_REF, ge::DataType::DT_RESOURCE},
  {tensorflow::DataType::DT_BFLOAT16_REF, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_UINT32_REF, ge::DataType::DT_UINT32},
  {tensorflow::DataType::DT_UINT64_REF, ge::DataType::DT_UINT64},
  {tensorflow::DataType::DT_INT64_REF, ge::DataType::DT_INT64},
  {tensorflow::DataType::DT_BOOL_REF, ge::DataType::DT_BOOL},
  {tensorflow::DataType::DT_HALF_REF, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_STRING_REF, ge::DataType::DT_STRING},
  {tensorflow::DataType::DT_VARIANT, ge::DataType::DT_VARIANT},
};
} // end

bool g_parse_root_graph = false;
bool g_geinit_fore_return_fail = false;
void SetParseRootGraph(bool is_root) {
  g_parse_root_graph = is_root;
}

static std::map<uint32_t, ge::Graph> graphs_map;
static std::atomic<bool> is_ge_init(false);
static std::atomic<bool> is_parser_init(false);

GE_FUNC_VISIBILITY Status GEInitialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  if (options.empty()) {
    return ge::FAILED;
  }
  if (g_geinit_fore_return_fail) {
    return ge::FAILED;
  }
  is_ge_init = true;
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY Status GEFinalize() {
  if (!is_ge_init) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status ParserInitialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  if (options.empty()) {
    return ge::FAILED;
  }
  is_parser_init = true;
  return ge::SUCCESS;
}

Status ParserFinalize() {
  if (!is_parser_init) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY ge::AscendString GEGetErrorMsgV2() { return ge::AscendString("ERROR");}
GE_FUNC_VISIBILITY ge::AscendString GEGetWarningMsgV2() { return ge::AscendString("WARNING"); }
GE_FUNC_VISIBILITY std::string GEGetErrorMsg() { return "ERROR";}

Session::Session(const std::map<string, string> &options) {}
Session::Session(const std::map<ge::AscendString, ge::AscendString> &options) {}

Session::~Session() {
  graphs_map.clear();
}

Status Session::RemoveGraph(uint32_t graphId) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    graphs_map.erase(ret);
    return ge::SUCCESS;
  }
  return ge::FAILED;
}

bool Session::IsGraphNeedRebuild(uint32_t graphId) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    return false;
  }
  return true;
}

Status Session::AddGraph(uint32_t graphId, const Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    return ge::SUCCESS;
  }
  graphs_map[graphId] = graph;
  return ge::SUCCESS;
}

Status Session::AddGraphWithCopy(uint32_t graphId, const Graph &graph, const std::map<AscendString, AscendString> &options) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    return ge::FAILED;
  }
  graphs_map[graphId] = graph;
  return ge::SUCCESS;
}

Status Session::BuildGraph(uint32_t graphId, const std::vector<ge::Tensor> &inputs) {
  return ge::SUCCESS;
}

RunGraphAsyncStub g_RunGraphAsyncStub = nullptr;
void RegRunGraphAsyncStub(RunGraphAsyncStub stub) {
  g_RunGraphAsyncStub = stub;
}

void ClearRegRunGraphAsyncStub() {
  g_RunGraphAsyncStub = nullptr;
}

Status Session::RunGraphAsync(uint32_t graphId, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback) {
  if (g_RunGraphAsyncStub != nullptr) {
    return g_RunGraphAsyncStub(graphId, inputs, callback);
  }
  std::vector<ge::Tensor> outputs;
  std::vector<uint8_t> data(4); // 初始化一个4字节大小的内存
  std::vector<int64_t> dims{};
  ge::Shape ge_shape(dims);
  ge::TensorDesc tensor_desc(ge_shape);
  outputs.push_back(ge::Tensor(tensor_desc, data.data(), data.size()));
  tensorflow::CallbackPack pack;
  pack.callback = callback;
  pack.ge_status = ge::SUCCESS;
  pack.outputs = outputs;
  tensorflow::CallbackExecutor::GetInstance().PushTask(pack);
  return ge::SUCCESS;
}

Status Session::BuildGraph(uint32_t graphId, const std::vector<InputTensorInfo> &inputs) {
  return ge::SUCCESS;
}

class ComputeGraph {
public:
  explicit ComputeGraph(const std::string &name) {}
  ~ComputeGraph() = default;
};

Graph::Graph(const std::string& grph) {}
Graph::Graph(char const* name) {}

void Graph::SetNeedIteration(bool need_iteration) {}

GNode::GNode() {}

std::vector<GNode> Graph::GetAllNodes() const {
  std::vector<GNode> res;
  GNode node;
  res.push_back(node);
  return res;
}

graphStatus aclgrphParseONNX(const char *model_file,
    const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph) {
  std::string model_(model_file);
  if(model_ == "no_model") {
    return FAILED;
  }
  return SUCCESS;
}

} // end ge

ge::Graph GeApiWrapper_CreateGraphFromComputeGraph(const ge::ComputeGraphPtr &compute_graph) {
  return ge::Graph();
}

ge::ComputeGraphPtr GeApiWrapper_MakeComputeGraphPtr(const char *graph_name) {
  return std::make_shared<ge::ComputeGraph>(graph_name);
}

size_t GeApiWrapper_GetAllNodesSize(const ge::ComputeGraphPtr &graph_ptr) {
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

void GeApiWrapper_RenameAllNodes(void *graph_ptr, const char *prefix) {
}

void GeApiWrapper_SetDomiContextTrainFlag(bool train_flag) {
}

ge::Status GeApiWrapper_ModelSaveToString(const ge::Graph &graph,
                                          const std::string &node_name,
                                          std::string &model_str) {
  return ge::SUCCESS;
}

ge::Status GeApiWrapper_ParseProtoWithSubgraph(const std::vector<ge::AscendString> &partitioned_serialized,
                                               const std::map<ge::AscendString, ge::AscendString> &const_value_map,
                                               domi::GetGraphCallbackV3 callback,
                                               ge::ComputeGraphPtr &graph) {
  return ge::SUCCESS;
}

ge::Status GeApiWrapper_GetGeDataTypeByTFType(const uint32_t type, ge::DataType &data_type) {
  auto search = ge::data_type_map.find(type);
  if (search != ge::data_type_map.end()) {
    data_type = search->second;
    return ge::SUCCESS;
  } else {
    data_type = ge::DataType::DT_UNDEFINED;
    return ge::FAILED;
  }
}

ge::Status GeApiWrapper_ParserFinalize() {
  return ge::ParserFinalize();
}

ge::Status GeApiWrapper_ParserInitialize(const std::map<ge::AscendString, ge::AscendString>& options) {
  return ge::ParserInitialize(options);
}

void GeApiWrapper_SetDomiFormatFromParserContext() {
}

ge::Status GeApiWrapper_InitRdmaPool(size_t size, rtMemType_t mem_type) {
  if (size == 0) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status GeApiWrapper_RdmaRemoteRegister(const std::vector<std::pair<uint64_t, uint64_t>> &var_info, rtMemType_t mem_type) {
  if (var_info.empty()) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status GeApiWrapper_MallocSharedMemory(const std::string &var_name, const std::vector<int64_t> &dims,
                                           ge::DataType data_type, uint64_t &dev_addr, uint64_t &memory_size) {
  if (var_name.empty()) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status GeApiWrapper_GetVarBaseAddrAndSize(const char *var_name, uint64_t &base_addr, uint64_t &var_size) {
  if (var_name == nullptr) {
    return ge::FAILED;
  }

  if (std::string(var_name) == "") {
    return ge::FAILED;
  }

  return ge::SUCCESS;
}


#ifdef __cplusplus
}
#endif
