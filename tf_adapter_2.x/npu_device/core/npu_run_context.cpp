/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_run_context.h"

namespace npu {
RunContextOptions &GetRunContextOptions() {
  static thread_local RunContextOptions run_context_options;
  return run_context_options;
}
}  // namespace npu

extern "C" {
void RunContextOptionsSetMemoryOptimizeOptions(const std::string &recompute) {
  npu::GetRunContextOptions().memory_optimize_options.recompute = recompute;
}
void CleanRunContextOptions() { npu::GetRunContextOptions().Clean(); }
}
