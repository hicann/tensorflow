/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_

#include "stub/defines.h"

namespace domi {
class GE_FUNC_VISIBILITY ModelParser {
 public:
  ModelParser() = default;
  ~ModelParser() = default;
  ge::DataType ConvertToGeDataType(const uint32_t type);
  Status ParseProtoWithSubgraph(const ge::AscendString &serialized_proto, GetGraphCallbackV3 callback,
                                ge::ComputeGraphPtr &graph);

  Status ParseProtoWithSubgraph(const std::vector<ge::AscendString> &partitioned_serialized,
                                const std::map<ge::AscendString, ge::AscendString> &const_value_map,
                                GetGraphCallbackV3 callback,
                                ge::ComputeGraphPtr &graph);
};
}  // namespace domi

#endif
