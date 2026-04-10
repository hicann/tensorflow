/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "external/aoe.h"
#include "external/aoe_errcodes.h"

namespace Aoe {
extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeFinalize() {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeCreateSession(uint64_t &sessionId) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeDestroySession(uint64_t sessionId) {
  if (sessionId >= 9999) {
    return Aoe::AOE_FAILURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetGeSession(uint64_t sessionId, ge::Session *geSession) {
  if (sessionId >= 9999) {
    return Aoe::AOE_FAILURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetDependGraphs(uint64_t sessionId, const std::vector<ge::Graph> &dependGraphs) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetTuningGraph(uint64_t sessionId, const ge::Graph &tuningGraph) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeTuningGraph(uint64_t sessionId,
                                    const std::map<ge::AscendString, ge::AscendString> &tuningOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetDependGraphsInputs(uint64_t sessionId,
                                              const std::vector<std::vector<ge::Tensor>> &inputs) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetTuningGraphInput(uint64_t sessionId, const std::vector<ge::Tensor> &input) {
  return Aoe::AOE_SUCCESS;
}
} // namespace Aoe
