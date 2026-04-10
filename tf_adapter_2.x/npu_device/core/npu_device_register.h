/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_DEVICE_REGISTER_H
#define NPU_DEVICE_CORE_NPU_DEVICE_REGISTER_H

#include <map>
#include <string>

#include "tensorflow/c/eager/c_api.h"

namespace npu {
std::string CreateDevice(TFE_Context *context, const char *name, int device_index,
                         const std::map<std::string, std::string> &global_options,
                         const std::map<std::string, std::string> &session_options);

void ReleaseDeviceResource();
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_DEVICE_REGISTER_H
