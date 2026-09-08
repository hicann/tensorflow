/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_global.h"
#include "npu_utils.h"

void SetCreateChannelWithCapacityStub(bool success);

int main() {
  tensorflow::DataType tf_type;
  aclDataType acl_type;
  aclFormat acl_format;
  if (npu::MapGeType2Tf(ge::DT_UNDEFINED, tf_type).ok() || npu::MapGeType2Acl(ge::DT_UNDEFINED, acl_type).ok() ||
      npu::MapGeFormat2Acl(ge::FORMAT_RESERVED, acl_format).ok()) {
    return 1;
  }

  char source = 0;
  if (npu::LoopCopy(nullptr, sizeof(source), &source, sizeof(source)).ok()) {
    return 2;
  }

  SetCreateChannelWithCapacityStub(false);
  const tensorflow::Status status =
      npu::global::GlobalHdcChannel::GetInstance().Create("st_unsupported_capacity", 1, {0});
  SetCreateChannelWithCapacityStub(true);
  npu::global::GlobalHdcChannel::GetInstance().Destroy("st_unsupported_capacity");
  return status.ok() ? 0 : 3;
}
