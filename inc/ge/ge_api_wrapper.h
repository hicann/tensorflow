/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_ADP_INC_GE_API_WRAPPER_H_
#define TENSORFLOW_ADP_INC_GE_API_WRAPPER_H_

#include <stddef.h>
#include <stdbool.h>
#include <map>
#include "ge_common/ge_api_types.h"
#include "runtime/mem.h"

namespace domi {
using GetGraphCallbackV3 = std::function<ge::AscendString(const ge::AscendString &subgraph_name)>;
} // namespace domi

namespace ge {
class Graph;
class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
} // namespace ge

ge::Graph GeApiWrapper_CreateGraphFromComputeGraph(const ge::ComputeGraphPtr &compute_graph);

size_t GeApiWrapper_GetComputeGraphInputSize(const ge::Graph &graph);

size_t GeApiWrapper_GetComputeGraphOutputSize(const ge::Graph &graph);

ge::ComputeGraphPtr GeApiWrapper_MakeComputeGraphPtr(const char *graph_name);

size_t GeApiWrapper_GetAllNodesSize(const ge::ComputeGraphPtr &graph_ptr);

#ifdef __cplusplus
extern "C" {
#endif

// Graph Helpers
void GeApiWrapper_RenameAllNodes(void *graph_ptr, const char *prefix);

void GeApiWrapper_SetDomiContextTrainFlag(bool train_flag);

ge::Status GeApiWrapper_ModelSaveToString(const ge::Graph &graph,
                                          const std::string &node_name,
                                          std::string &model_str);

ge::Status GeApiWrapper_ParseProtoWithSubgraph(const std::vector<ge::AscendString> &partitioned_serialized,
                                               const std::map<ge::AscendString, ge::AscendString> &const_value_map,
                                               domi::GetGraphCallbackV3 callback,
                                               ge::ComputeGraphPtr &graph);

ge::Status GeApiWrapper_GetGeDataTypeByTFType(const uint32_t type, ge::DataType &data_type);

ge::Status GeApiWrapper_ParserFinalize();

ge::Status GeApiWrapper_ParserInitialize(const std::map<ge::AscendString, ge::AscendString>& options);

void GeApiWrapper_SetDomiFormatFromParserContext();

ge::Status GeApiWrapper_InitRdmaPool(size_t size, rtMemType_t mem_type);

ge::Status GeApiWrapper_RdmaRemoteRegister(const std::vector<std::pair<uint64_t, uint64_t>> &var_info, rtMemType_t mem_type);

ge::Status GeApiWrapper_GetVarBaseAddrAndSize(const char *var_name, uint64_t &base_addr, uint64_t &var_size);

ge::Status GeApiWrapper_MallocSharedMemory(const std::string &var_name, const std::vector<int64_t> &dims,
                                           ge::DataType data_type, uint64_t &dev_addr, uint64_t &memory_size);

#ifdef __cplusplus
}
#endif

#endif // TENSORFLOW_ADP_INC_GE_API_WRAPPER_H_
