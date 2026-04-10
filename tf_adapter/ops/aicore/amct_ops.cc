/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
REGISTER_OP("AscendQuant")
  .Attr("T: {float16, float32, float64}")
  .Attr("dst_type: {'INT4', 'INT8'} = 'INT8'")
  .Attr("quant_bits: int = 8")
  .Attr("scale: float")
  .Attr("offset: float")
  .Input("x: T")
  .Output("y: T")
  .SetIsStateful()
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("AscendWeightQuant")
  .Attr("T: {float16, float32, float64}")
  .Attr("dst_type: {'INT4', 'INT8'} = 'INT8'")
  .Input("x: int8")
  .Input("offset_w: int8")
  .Output("y: T")
  .SetIsStateful()
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("AscendDequant")
  .Attr("T: {float16, float32, float64}")
  .Attr("ksize: list(int)")
  .Attr("data_format: string = 'NHWC'")
  .Input("x: T")
  .Input("deq_scale: uint64")
  .Output("y: T")
  .SetIsStateful()
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("AscendAntiQuant")
  .Attr("T: {float16, float32, float64}")
  .Attr("scale: float")
  .Attr("offset: float")
  .Input("x: T")
  .Output("y: T")
  .SetIsStateful()
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });
}  // namespace tensorflow
