/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/common/adapter_logger.h"
#include "toolchain/slog.h"
#include "base/plog.h"

namespace npu {
AdapterLogger::~AdapterLogger() {
  int32_t modeule = FMK_MODULE_NAME;
  int32_t log_level = severity_;
  if (severity_ == ADP_RUN_INFO) {
    // ADP_RUN_INFO and ADP_INFO use same log level,but module is not same
    modeule = static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(FMK_MODULE_NAME));
    log_level = ADP_INFO;
  }
  if (severity_ == ADP_FATAL) {
    DlogSub(modeule, ADP_MODULE_NAME, ADP_ERROR, "%s", str().c_str());
    (void) DlogReportFinalize();
  } else {
    DlogSub(modeule, ADP_MODULE_NAME, log_level, "%s", str().c_str());
  }
}
}  // namespace npu
