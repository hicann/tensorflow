/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_DEBUG_GE_UTIL_H_
#define COMMON_GRAPH_DEBUG_GE_UTIL_H_

#include <iostream>

#include "graph/types.h"
#include "ge/ge_api.h"
#include "ge_common/ge_api_types.h"
#include "graph/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace ge {
extern bool g_parse_root_graph;
extern bool g_geinit_fore_return_fail;
void SetParseRootGraph(bool is_root);

using RunGraphWithStreamAsyncStub = std::function<Status(uint32_t, void *, const std::vector<Tensor>&, std::vector<Tensor>&)>;
void RegRunGraphWithStreamAsyncStub(RunGraphWithStreamAsyncStub stub);

using RunGraphStub = std::function<Status(uint32_t, const std::vector<Tensor>&, std::vector<Tensor>&)>;
void RegRunGraphStub(RunGraphStub stub);

using RunGraphAsyncStub = std::function<Status(uint32_t, const std::vector<Tensor>&, RunAsyncCallback)>;
void RegRunGraphAsyncStub(RunGraphAsyncStub stub);
void ClearRegRunGraphAsyncStub();
void SetCustomPathStub(std::string path);
const char* GetCustomPathStub();
}  // namespace ge
#endif  // COMMON_GRAPH_DEBUG_GE_UTIL_H_
