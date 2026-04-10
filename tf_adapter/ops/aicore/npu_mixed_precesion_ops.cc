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
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::UnchangedShape;
// Mixed-precisions training
REGISTER_OP("NpuAllocFloatStatus")
    .Output("float_status: float")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->Vector(1));
      return Status::OK();
    })
    .Doc(R"doc(
    Allocate the float status tensor for getting float status from scalar buffer.

    Arguments
        inputs: No inputs.

    Output
        output: One float element tensor.
    )doc")
    .SetIsStateful();

REGISTER_OP("NpuGetFloatStatus")
    .Input("input_float: float")
    .Output("float_status: float")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
    Allocate the float status tensor for getting float status from scalar buffer.

    Arguments
        inputs: The allocated input float status tensor.

    Output
        output: The one float status element tensor.
    )doc")
    .SetIsStateful();

REGISTER_OP("NpuGetFloatStatusV2")
    .Output("data: int32")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        std::vector<DimensionHandle> output_dims;
        output_dims.emplace_back(c->MakeDim(8));
        auto output_shape = c->MakeShape(output_dims);
        c->set_output(0, output_shape);
        return Status::OK();
    });


REGISTER_OP("NpuClearFloatStatus")
    .Input("float_status: float")
    .Output("cleared_float_status: float")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
    Clear the float status in the scalar buffer.

    Arguments
        inputs: The float status tensor.

    Output
        output: The float element tensor set to zero.
    )doc")
    .SetIsStateful();

REGISTER_OP("NpuClearFloatStatusV2")
    .Doc(R"doc(
    Set the value of global workspace to 0.
    )doc")
    .SetIsStateful();
}  // namespace tensorflow
