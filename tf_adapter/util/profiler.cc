/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiler.h"
#include "tf_adapter/common/adapter_logger.h"
#include "npu_attrs.h"

namespace {
constexpr uint64_t Level_none = 0;
constexpr uint64_t Level0 = ACL_PROF_TASK_TIME_L0 | ACL_PROF_ACL_API;
constexpr uint64_t Level1 = ACL_PROF_TASK_TIME | ACL_PROF_ACL_API | ACL_PROF_HCCL_TRACE | ACL_PROF_AICORE_METRICS;
constexpr uint64_t Level2 = Level1 | ACL_PROF_RUNTIME_API | ACL_PROF_AICPU;
std::map<std::string, aclprofAicoreMetrics> kNpuMetricsMap = {
    {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
    {"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
    {"Memory", ACL_AICORE_MEMORY_BANDWIDTH},
    {"MemoryL0", ACL_AICORE_L0B_AND_WIDTH},
    {"ResourceConflictRatio", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
    {"MemoryUB", ACL_AICORE_MEMORY_UB},
    {"L2Cache", ACL_AICORE_L2_CACHE}
};
std::map<std::string, uint64_t> kProfilerLevelMap = {
    {"L0", Level0},
    {"L1", Level1},
    {"L2", Level2}
};
}
namespace tensorflow {
Profiler &Profiler::GetInstance() {
  static Profiler instance;
  return instance;
}

Status Profiler::GetLevel(const std::string &level) {
  const auto level_iter = kProfilerLevelMap.find(level);
  if (level_iter != kProfilerLevelMap.cend()) {
    level_ = level_iter->second;
    ADP_LOG(INFO) << "Profiler level: " << level;
    return Status::OK();
  }
  std::string error_msg = "Profiler options: level cannot set to: " + level + ", should set level to [";
  size_t i = 0UL;
  for (const auto &iter : kProfilerLevelMap) {
    error_msg += iter.first;
    i++;
    if (i < kProfilerLevelMap.size()) {
      error_msg += ", ";
    }
  }
  error_msg += "].";
  ADP_LOG(ERROR) << error_msg;
  return errors::InvalidArgument(error_msg);
}

Status Profiler::GetAicMetrics(const std::string &aic_metrics) {
  if ((level_ != Level0) && (aic_metrics.empty())) {
    ADP_LOG(INFO) << "Profiler aic_metrics is empty, set default: PipeUtilization";
    aic_metrics_ = ACL_AICORE_PIPE_UTILIZATION;
    return Status::OK();
  }

  if (level_ == Level0) {
    if (!aic_metrics.empty()) {
      return errors::InvalidArgument("Please use L1 or L2 if you want to collect aic metrics!");
    }
    aic_metrics_ = ACL_AICORE_NONE;
    return Status::OK();
  }

  const auto metrics_iter = kNpuMetricsMap.find(aic_metrics);
  if (metrics_iter != kNpuMetricsMap.cend()) {
    ADP_LOG(INFO) << "Profiler aic_metrics: " << aic_metrics;
    aic_metrics_ = metrics_iter->second;
    return Status::OK();
  }
  std::string error_msg = "Profiler options: aic_metrics cannot set to: " +
      aic_metrics + ", should set aic_metrics to [";
  size_t i = 0UL;
  for (const auto &iter : kNpuMetricsMap) {
    error_msg += iter.first;
    i++;
    if (i < kNpuMetricsMap.size()) {
      error_msg += ", ";
    }
  }
  error_msg += "].";
  ADP_LOG(ERROR) << error_msg;
  return errors::InvalidArgument(error_msg);
}

Status Profiler::Enable(const std::string &level,
    const std::string &aic_metrics, const std::string &output_path) {
  if (enable_flag_) {
    return errors::Internal("Not support nested call 'profiler.Profiler'.");
  }
  ADP_LOG(INFO) << "Enable Profiler";
  auto ret = GetLevel(level);
  if (!ret.ok()) {
    return ret;
  }
  ret = GetAicMetrics(aic_metrics);
  if (!ret.ok()) {
    return ret;
  }
  // 输出path不能为空，否则profiling报错，如果path为空，则落盘在当前目录下
  output_path_ = output_path.empty() ? "./" : output_path;
  enable_flag_ = true;
  return Status::OK();
}

Status Profiler::Start() {
  if (has_start_) {
    return Status::OK();
  }
  ADP_LOG(INFO) << "Profiler Start";
  has_start_ = true;
  if (aclprofInit(output_path_.c_str(), output_path_.size()) != ACL_ERROR_NONE) {
    return errors::Internal("Call aclprofInit failed");
  }
  uint32_t device_id = 0U;
  auto ret = GetDeviceID(device_id);
  if (!ret.ok()) {
    return ret;
  }
  if (prof_config_ != nullptr) {
    (void)aclprofDestroyConfig(prof_config_);
    prof_config_ = nullptr;
    return errors::Internal("Prof config has been create, check if destroy config failed.");
  }
  prof_config_ = aclprofCreateConfig(&device_id, 1U, aic_metrics_, nullptr, level_);
  if (prof_config_ == nullptr) {
    return errors::Internal("Create prof config failed.");
  }
  if (aclprofStart(prof_config_) != ACL_ERROR_NONE) {
    return errors::Internal("Call aclprofStart failed");
  }
  return Status::OK();
}

Status Profiler::Stop() {
  if (!has_start_) {
    return Status::OK();
  }
  has_start_ = false;
  ADP_LOG(INFO) << "Profiler Stop";
  auto ret_stop = aclprofStop(prof_config_);
  auto ret_destroy_config = aclprofDestroyConfig(prof_config_);
  prof_config_ = nullptr;
  auto ret_finalize = aclprofFinalize();
  if (ret_stop != ACL_ERROR_NONE) {
    return errors::Internal("Call aclprofStop failed");
  }
  if (ret_destroy_config != ACL_ERROR_NONE) {
    return errors::Internal("Call aclprofDestroyConfig failed");
  }
  if (ret_finalize != ACL_ERROR_NONE) {
    return errors::Internal("Call aclprofFinalize failed");
  }
  return Status::OK();
}

void Profiler::Disable() {
  ADP_LOG(INFO) << "Disable Profiler";
  enable_flag_ = false;
}
}
