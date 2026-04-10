/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_RUN_CONTEXT_H
#define NPU_DEVICE_CORE_NPU_RUN_CONTEXT_H

#include <string>
#include <map>

namespace npu {
struct MemoryOptimizeOptions {
  MemoryOptimizeOptions() {}
  void Clean() { recompute.clear(); }
  std::string recompute;
};

struct RunContextOptions {
  MemoryOptimizeOptions memory_optimize_options;
  void Clean() { memory_optimize_options.Clean(); }
  std::map<std::string, std::string> GetGraphOptions() {
    std::map<std::string, std::string> kOptions = {
      {"ge.recompute", memory_optimize_options.recompute}};
    return kOptions;
  }
};

RunContextOptions &GetRunContextOptions();
}  // namespace npu

extern "C" {
extern void RunContextOptionsSetMemoryOptimizeOptions(const std::string &recompute);
extern void CleanRunContextOptions();
}

#endif  // NPU_DEVICE_CORE_NPU_RUN_CONTEXT_H
