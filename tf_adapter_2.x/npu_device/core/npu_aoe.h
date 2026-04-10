/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_AOE_H
#define NPU_DEVICE_CORE_NPU_AOE_H

#include <map>
#include "external/aoe.h"
#include "external/aoe_errcodes.h"
#include "npu_device.h"

namespace npu {
using SessionId = uint64_t;
using AoeStatus = int32_t;
using AoeInitializeFunc = AoeStatus (*)(const std::map<ge::AscendString, ge::AscendString> &);
using AoeFinalizeFunc = AoeStatus (*)();
using AoeCreateSessionFunc = AoeStatus (*)(SessionId &);
using AoeDestroySessionFunc = AoeStatus (*)(SessionId);
using AoeSetGeSessionFunc = AoeStatus (*)(SessionId, ge::Session *);
using AoeSetDependGraphFunc = AoeStatus (*)(SessionId, const std::vector<ge::Graph> &);
using AoeSetDependGraphsInputsFunc = AoeStatus (*)(SessionId, const std::vector<std::vector<ge::Tensor>> &);
using AoeSetTuningGraphInputFunc = AoeStatus (*)(SessionId, const std::vector<ge::Tensor> &);
using AoeSetTuningGraphFunc = AoeStatus (*)(SessionId, const ge::Graph &);
using AoeTuningGraphFunc = AoeStatus (*)(SessionId, const std::map<ge::AscendString, ge::AscendString> &);

struct AoeFunc {
  AoeInitializeFunc aoe_initialize = nullptr;
  AoeFinalizeFunc aoe_finalize = nullptr;
  AoeCreateSessionFunc aoe_create_session = nullptr;
  AoeDestroySessionFunc aoe_destroy_session = nullptr;
  AoeSetGeSessionFunc aoe_set_gesession = nullptr;
  AoeSetDependGraphFunc aoe_set_dependgraphs = nullptr;
  AoeSetTuningGraphFunc aoe_set_tuninggraph = nullptr;
  AoeTuningGraphFunc aoe_tuning_graph = nullptr;
  AoeSetDependGraphsInputsFunc aoe_set_depend_graphs_inputs = nullptr;
  AoeSetTuningGraphInputFunc aoe_set_tuning_graph_input = nullptr;
};

class NpuAoe {
 public:
  NpuAoe() = default;
  ~NpuAoe();

  static NpuAoe &GetInstance();
  tensorflow::Status AoeTuningInitialize(const std::string &work_path, const std::string &job_type);
  tensorflow::Status RunAoeTuning(NpuDevice &device, TFE_Context *context, bool need_build, uint64_t graph_id,
                                  const std::string &name, const tensorflow::GraphDef &graph_def,
                                  std::vector<TFE_TensorHandle *> &inputs);
  tensorflow::Status AoeTuningFinalize();

  NpuAoe(const NpuAoe&) = delete;
  NpuAoe(NpuAoe &&) = delete;
  NpuAoe& operator=(const NpuAoe&) = delete;
  NpuAoe& operator=(NpuAoe &&) = delete;

 private:
  tensorflow::Status LoadAoeFunc();

  AoeFunc aoe_func_;
  void *handle_ = nullptr;
  int64_t exec_num_ = 0;
  std::map<uint64_t, ge::Graph> ge_graph_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_AOE_H
