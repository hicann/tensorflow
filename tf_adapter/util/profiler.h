/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_UTILS_PROFILER_H_
#define TENSORFLOW_UTILS_PROFILER_H_
#include "acl/acl_prof.h"
#include <string>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
  class Profiler {
   public:
    static Profiler &GetInstance();
    Status Enable(const std::string &level,
        const std::string &aic_metrics, const std::string &output_path);
    Status Start();
    Status Stop();
    void Disable();
    bool IsEnabled() { return enable_flag_; }
   private:
    Profiler() = default;
    explicit Profiler(const Profiler &obj) = delete;
    Profiler& operator=(const Profiler &obj) = delete;
    explicit Profiler(Profiler &&obj) = delete;
    Profiler& operator=(Profiler &&obj) = delete;
    Status GetLevel(const std::string &level);
    Status GetAicMetrics(const std::string &aic_metrics);
    aclprofConfig *prof_config_{nullptr};
    uint64_t level_{0UL};
    aclprofAicoreMetrics aic_metrics_{ACL_AICORE_NONE};
    bool enable_flag_{false};
    bool has_start_{false};
    std::string output_path_;
  };
}

#endif
