/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tdt/tdt_host_interface.h"

namespace tdt {
int32_t TdtInFeedInit(uint32_t deviceId) { return 0; }

int32_t TdtOutFeedInit(uint32_t deviceId) { return 0; }

int32_t TdtInFeedDestroy(uint32_t deviceId) { return 0; }

int32_t TdtOutFeedDestroy() { return 0; }

int32_t TdtHostPreparePopData() { return 0; }

int32_t TdtHostPopData(const std::string &channelName, std::vector<DataItem> &item) { return 0; }

int32_t TdtHostStop(const std::string &channelName) { return 0; }

int32_t TdtHostPushData(const std::string &channelName, const std::vector<DataItem> &item, uint32_t deviceId) { return 0; }
}
