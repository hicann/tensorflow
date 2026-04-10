/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_ENV_H
#define NPU_DEVICE_CORE_NPU_ENV_H

#include "tensorflow/core/util/env_var.h"

const static bool kDumpExecutionDetail = []() -> bool {
  bool dump_execute_detail = false;
  (void)tensorflow::ReadBoolFromEnvVar("NPU_DEBUG", false, &dump_execute_detail);
  return dump_execute_detail;
}();

const static bool kDumpGraph = []() -> bool {
  bool dump_graph = false;
  (void)tensorflow::ReadBoolFromEnvVar("NPU_DUMP_GRAPH", false, &dump_graph);
  return dump_graph;
}();

const static bool kPerfEnabled = []() -> bool {
  bool perf_enabled = false;
  (void)tensorflow::ReadBoolFromEnvVar("NPU_ENABLE_PERF", false, &perf_enabled);
  return perf_enabled;
}();

const static bool kGraphEngineGreedyMemory = []() -> bool {
  tensorflow::int64 graph_engine_greedy_memory = 0;
  (void)tensorflow::ReadInt64FromEnvVar("GE_USE_STATIC_MEMORY", 0, &graph_engine_greedy_memory);
  return graph_engine_greedy_memory == 1;
}();

const static std::string kOppInstallPath = []() -> std::string {
  std::string opp_install_path;
  (void)tensorflow::ReadStringFromEnvVar("ASCEND_OPP_PATH", "", &opp_install_path);
  return opp_install_path;
}();

const static std::string kAoeMode = []() -> std::string {
  std::string aoe_mode;
  (void)tensorflow::ReadStringFromEnvVar("AOE_MODE", "", &aoe_mode);
  return aoe_mode;
}();

#endif  // NPU_DEVICE_CORE_NPU_ENV_H
