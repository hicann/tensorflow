/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiler_interface.h"
#include "profiler.h"

const std::string ProfilerStart(const std::string &level,
    const std::string &aic_metrics,
    const std::string &output_path) {
  return tensorflow::Profiler::GetInstance().Enable(level, aic_metrics, output_path).error_message();
}

const std::string ProfilerStop() {
  tensorflow::Profiler::GetInstance().Disable();
  return tensorflow::Profiler::GetInstance().Stop().error_message();
}
