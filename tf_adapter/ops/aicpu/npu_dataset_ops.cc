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
REGISTER_OP("QueueDataset")
  .Input("input_dataset: variant")
  .Attr("sourcedata: string")
  .Output("handle: variant")
  .SetIsStateful()
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("HostQueueDataset")
  .Input("geop_dataset: variant")
  .Input("input_dataset: variant")
  .Attr("channel_name: string")
  .Output("handle: variant")
  .SetIsStateful()
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("DeviceQueueDataset")
  .Attr("channel_name: string")
  .Output("handle: variant")
  .SetIsStateful()
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("GEOPDataset")
  .Output("handle: variant")
  .Attr("f: func")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DPGroupDataset")
  .Input("input_datasets: N * variant")
  .Output("handle: variant")
  .Attr("N: int >= 0")
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AdpGetNext")
  .Output("components: output_types")
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .Attr("queue_name: string")
  .SetIsStateful()
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("GetNext")
  .Output("components: output_types")
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .Attr("channel_name: string")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    std::vector<PartialTensorShape> output_shapes;
    TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
    if (output_shapes.size() != static_cast<size_t>(c->num_outputs())) {
      return errors::InvalidArgument(
        "output_shapes must be the same length as output_types, output_shapes is ",
        output_shapes.size(), ", but output_types is ", c->num_outputs());
    }
    for (size_t i = 0UL; i < output_shapes.size(); i++) {
      shape_inference::ShapeHandle output_shape_handle;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        output_shapes[i], &output_shape_handle));
      c->set_output(static_cast<int>(i), output_shape_handle);
    }
    return Status::OK();
  });

REGISTER_OP("DynamicGetNextV2")
  .Output("components: output_types")
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .Attr("channel_name: string")
  .SetIsStateful()
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("GetNextFromQueue")
  .Input("data: uint8")
  .Output("components: output_types")
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .SetIsStateful()
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("QueueData")
  .Output("output: T")
  .Attr("index: int >= 0")
  .Attr("T: type")
  .Attr("queue_name: string")
  .Attr("output_types: list(type) >= 1")
  .Attr("output_shapes: list(shape) >= 1")
  .SetIsStateful()
  .SetShapeFn(tensorflow::shape_inference::ScalarShape);
}  // namespace tensorflow
