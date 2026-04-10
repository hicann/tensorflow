/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_LOGGER_H
#define NPU_DEVICE_CORE_NPU_LOGGER_H

#include "tensorflow/core/platform/env.h"

#include "npu_env.h"
#include "npu_hdc.h"

#define DLOG() \
  if (kDumpExecutionDetail) LOG(INFO)

namespace npu {
// 日志适配层，需要对接slog，当前未使用，复用的tensorflow
class Logger : public std::basic_ostringstream<char> {
 public:
  Logger(const char *f, int line) { *this << f << ":" << line << " "; }
  ~Logger() override { std::cerr << str() << std::endl; }
};

class Timer : public std::basic_ostringstream<char> {
 public:
  template <typename... Args>
  explicit Timer(const Args... args) {
    *this << tensorflow::strings::StrCat(args...) << " cost ";
  };
  ~Timer() override = default;
  void Start() {
    if (TF_PREDICT_FALSE(kPerfEnabled)) {
      start_ = tensorflow::Env::Default()->NowMicros();
    }
    started_ = true;
  }
  void Stop() {
    if (started_ && TF_PREDICT_FALSE(kPerfEnabled)) {
      constexpr uint64_t kMicrosToMillis = 1000ULL;
      *this << (tensorflow::Env::Default()->NowMicros() - start_) / kMicrosToMillis << " ms";
      LOG(INFO) << str();
    }
    started_ = false;
  }

 private:
  uint64_t start_{0};
  bool started_{false};
};

class NpuStdoutReceiver {
 public:
  explicit NpuStdoutReceiver(uint32_t device_id) : device_id_(device_id){};

  tensorflow::Status Start();

  tensorflow::Status Stop();

 private:
  uint32_t device_id_;
  std::mutex mu_;
  std::thread thread_;
  std::shared_ptr<HdcChannel> channel_ TF_GUARDED_BY(mu_);
  bool started_{false};
  std::atomic_bool stopping_{false};
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_LOGGER_H
