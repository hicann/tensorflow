/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_ADP_LOGGER_H
#define TENSORFLOW_ADP_LOGGER_H

#include <sstream>
#include "mmpa/mmpa_api.h"

#define FMK_MODULE_NAME static_cast<int>(FMK)

#define LOG_DEPRECATED_WITH_REPLACEMENT(old, replacement)                                            \
  do {                                                                                               \
    static bool first_warning_##old = true;                                                          \
    if (first_warning_##old) {                                                                       \
      LOG(WARNING) << "[warning][tf_adapter] Option \'" #old "\' is deprecated and will be removed " \
                      "in future version. Please use \'" #replacement "\' instead.";                 \
      first_warning_##old = false;                                                                   \
    }                                                                                                \
  } while (false)

#define LOG_DEPRECATED(old)                                                                          \
  do {                                                                                               \
    static bool first_warning_##old = true;                                                          \
    if (first_warning_##old) {                                                                       \
      LOG(WARNING) << "[warning][tf_adapter] Option \'" #old "\' is deprecated and will be removed " \
                      "in future version. Please do not configure this option in the future.";       \
      first_warning_##old = false;                                                                   \
    }                                                                                                \
  } while (false)

namespace npu {
constexpr const char *ADP_MODULE_NAME = "TF_ADAPTER";
const int ADP_DEBUG = 0;
const int ADP_INFO = 1;
const int ADP_WARNING = 2;
const int ADP_ERROR = 3;
const int ADP_RUN_INFO = 4;
const int ADP_FATAL = 32;

class AdapterLogger : public std::basic_ostringstream<char> {
 public:
  AdapterLogger(const char *fname, int line, int severity) : severity_(severity) {
    *this << " [" << fname << ":" << line << "]" << GetTid() << " ";
  }
  ~AdapterLogger() override;

 private:
  mmPid_t GetTid() const {
    static const thread_local mmPid_t tid = static_cast<mmPid_t>(mmGetTid());
    return tid;
  }
  int severity_;
};
}  // namespace npu

#define ADP_LOG_INFO npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_INFO)
#define ADP_LOG_WARNING npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_WARNING)
#define ADP_LOG_ERROR npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_ERROR)
#define ADP_LOG_EVENT npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_RUN_INFO)
#define ADP_LOG_DEBUG npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_DEBUG)
#define ADP_LOG_FATAL npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_FATAL)

#define ADP_LOG(LEVEL) ADP_LOG_##LEVEL
#endif
