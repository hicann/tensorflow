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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("InitEmbeddingHashmapV2")
  .Input("table_id: int32")
  .Output("table_handle: int64")
  .Attr("bucket_size: int")
  .Attr("load_factor: int")
  .Attr("embedding_dim: int")
  .Attr("dtype: type = DT_FLOAT")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("DeinitEmbeddingHashmapV2")
  .Input("table_id: int32")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("TableToResourceV2")
  .Input("table_id: int32")
  .Output("table_handle: int64")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("EmbeddingHashmapExport")
  .Input("file_path: string")
  .Input("table_ids: int32")
  .Input("table_names: string")
  .Input("global_step: TStep")
  .Input("keys: num * int64")
  .Input("counters: num * uint64")
  .Input("filter_flags: num * uint8")
  .Input("values: num * float32")
  .Attr("num: int >= 1")
  .Attr("TStep: {int32, int64}")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingHashmapSize")
  .Input("table_ids: int32")
  .Output("table_sizes: int64")
  .Attr("filter_export_flag: bool = false")
  .Attr("export_mode: {'all', 'old', 'new', 'specifiednew'} = 'all'")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("EmbeddingHashmapFileSize")
  .Input("file_path: string")
  .Input("table_ids: int32")
  .Input("table_names: string")
  .Input("global_step: TStep")
  .Output("table_sizes: int64")
  .Attr("embedding_dims: list(int)")
  .Attr("TStep: {int32, int64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(1));
    return Status::OK();
  });

REGISTER_OP("EmbeddingHashmapImport")
  .Input("file_path: string")
  .Input("table_ids: int32")
  .Input("table_sizes: int64")
  .Input("table_names: string")
  .Input("global_step: TStep")
  .Output("keys: num * int64")
  .Output("counters: num * uint64")
  .Output("filter_flags: num * uint8")
  .Output("values: num * float32")
  .Attr("embedding_dims: list(int)")
  .Attr("num: int >= 1")
  .Attr("TStep: {int32, int64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    int64 num = 0;
    c->GetAttr("num", &num);
    for (int64_t i = 0; i < num; ++i) {
      c->set_output(i, c->Vector(c->UnknownDim()));
      c->set_output(i + num, c->Vector(c->UnknownDim()));
      c->set_output(i + 2 * num, c->Vector(c->UnknownDim()));
      c->set_output(i + 3 * num, c->Vector(c->UnknownDim()));
    }
    return Status::OK();
  });
}  // namespace tensorflow
