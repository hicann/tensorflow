/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
REGISTER_OP("SendH2D")
  .Input("inputs: Tin")
  .Attr("channel_name: string")
  .Attr("device_ids: list(int)")
  .Attr(
    "Tin: list(type) = [DT_FLOAT, DT_HALF, DT_INT8, DT_INT32, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, "
    "DT_INT64, DT_UINT64, DT_DOUBLE, DT_BOOL, DT_STRING]")
  .SetIsStateful();

REGISTER_OP("IteratorH2D")
  .Input("input: resource")
  .Input("nums: int64")
  .Attr("channel_name: string")
  .Attr("device_ids: list(int)")
  .SetIsStateful();

REGISTER_OP("NpuCall")
  .Input("args: Tin")
  .Output("output: Tout")
  .Attr("Tin: list(type) >= 0")
  .Attr("Tout: list(type) >= 0")
  .Attr("f: func")
  .Attr("device: int")
  .SetIsStateful()
  .SetShapeFn(shape_inference::UnknownShape);
}  // namespace tensorflow
