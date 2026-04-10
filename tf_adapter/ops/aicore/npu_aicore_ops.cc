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

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

namespace {
REGISTER_OP("FastGelu")
  .Input("features: T")
  .Output("activations: T")
  .Attr("T: realnumbertype")
  .SetShapeFn(tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("FastGeluV2")
  .Input("features: T")
  .Output("activations: T")
  .Attr("T: realnumbertype")
  .SetShapeFn(tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("FastGeluGrad")
  .Input("gradients: T")
  .Input("features: T")
  .Output("backprops: T")
  .Attr("T: realnumbertype")
  .SetShapeFn(tensorflow::shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("EmbeddingHashTableImport")
  .Input("table_handles: int64")
  .Input("embedding_dims: int64")
  .Input("bucket_sizes: int64")
  .Input("keys: num * int64")
  .Input("counters: num * uint64")
  .Input("filter_flags: num * uint8")
  .Input("values: num * float32")
  .Attr("num: int >= 1")
  .SetShapeFn(tensorflow::shape_inference::NoOutputs);

REGISTER_OP("EmbeddingHashTableExport")
  .Input("table_handles: int64")
  .Input("table_sizes: int64")
  .Input("embedding_dims: int64")
  .Input("bucket_sizes: int64")
  .Output("keys: num * int64")
  .Output("counters: num * uint64")
  .Output("filter_flags: num * uint8")
  .Output("values: num * float")
  .Attr("export_mode: string = 'all'")
  .Attr("filter_export_flag: bool = false")
  .Attr("num: int >= 1")
  .SetIsStateful()
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

REGISTER_OP("EmbeddingHashTableApplyAdamW")
  .Input("table_handle: int64")
  .Input("keys: int64")
  .Input("m: Ref(T)")
  .Input("v: Ref(T)")
  .Input("beta1_power: Ref(T)")
  .Input("beta2_power: Ref(T)")
  .Input("lr: T")
  .Input("weight_decay: T")
  .Input("beta1: T")
  .Input("beta2: T")
  .Input("epsilon: T")
  .Input("grad: T")
  .Input("max_grad_norm: Ref(T)")
  .Attr("embedding_dim: int")
  .Attr("bucket_size: int")
  .Attr("amsgrad: bool = false")
  .Attr("maximize: bool = false")
  .Attr("T: {float16, float32}")
  .SetShapeFn(tensorflow::shape_inference::NoOutputs);

REGISTER_OP("StatelessRandomChoiceWithMask")
  .Input("x: bool")
  .Input("count: int32")
  .Input("seed: int64")
  .Input("offset: int64")
  .Output("y: int32")
  .Output("mask: bool")
  .SetShapeFn([](InferenceContext *c) {
    auto rank = InferenceContext::Rank(c->input(0));
    c->set_output(0, c->MakeShape({2, rank}));
    c->set_output(1, c->MakeShape({2}));
    return Status::OK();
  });

REGISTER_OP("DynamicGruV2")
  .Input("x: T")
  .Input("weight_input: T")
  .Input("weight_hidden: T")
  .Input("bias_input: T")
  .Input("bias_hidden: T")
  .Input("seq_length: int32")
  .Input("init_h: T")
  .Output("y: T")
  .Output("output_h: T")
  .Output("update: T")
  .Output("reset: T")
  .Output("new: T")
  .Output("hidden_new: T")
  .Attr("T: {float16, float32}")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("activation: string")
  .Attr("gate_order: string")
  .Attr("reset_after: bool = true")
  .Attr("is_training: bool = true")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto weight_hidden_shape = c->input(2);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto hidden_size = c->Dim(weight_hidden_shape, 0);
    int32_t num_proj = 0;
    TF_RETURN_IF_ERROR(c->GetAttr("num_proj", &num_proj));
    ShapeHandle output_y_shape;
    if (num_proj == 0) {
      output_y_shape = c->MakeShape({num_step, batch_size, hidden_size});
    } else {
      auto num_proj_size = c->MakeDim(shape_inference::DimensionOrConstant(num_proj));
      DimensionHandle output_hidden_size;
      TF_RETURN_IF_ERROR(c->Min(num_proj_size, hidden_size, &output_hidden_size));
      output_y_shape = c->MakeShape({num_step, batch_size, output_hidden_size});
    }
    auto output_h_shape = c->MakeShape({num_step, batch_size, hidden_size});
    c->set_output(0, output_y_shape);
    c->set_output(1, output_h_shape);
    c->set_output(2, c->UnknownShape());
    c->set_output(3, c->UnknownShape());
    c->set_output(4, c->UnknownShape());
    c->set_output(5, c->UnknownShape());
    return Status::OK();
  });

REGISTER_OP("StatelessDropout")
  .Input("x: T")
  .Input("noise_shape: int64")
  .Input("p: T")
  .Input("seed: int64")
  .Input("offset: int64")
  .Output("y: T")
  .Attr("T: {float16, float32, bfloat16}")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("DynamicGruV2Grad")
  .Input("x: T")
  .Input("weight_input: T")
  .Input("weight_hidden: T")
  .Input("y: T")
  .Input("init_h: T")
  .Input("h: T")
  .Input("dy: T")
  .Input("dh: T")
  .Input("update: T")
  .Input("reset: T")
  .Input("new: T")
  .Input("hidden_new: T")
  .Input("seq_length: int32")
  .Output("dw_input: T")
  .Output("dw_hidden: T")
  .Output("db_input: T")
  .Output("db_hidden: T")
  .Output("dx: T")
  .Output("dh_prev: T")
  .Attr("T: {float16, float32}")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("gate_order: string")
  .Attr("reset_after: bool = true")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto weight_hidden_shape = c->input(2);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto input_size = c->Dim(input_shape, 2);
    auto hidden_size = c->Dim(weight_hidden_shape, 0);
    auto hidden_size_1 = c->Dim(weight_hidden_shape, 1);
    auto output_dw_input_shape = c->MakeShape({input_size, hidden_size_1});
    auto output_dw_hidden_shape = c->MakeShape({hidden_size, hidden_size_1});
    auto output_db_input_shape = c->MakeShape({hidden_size_1});
    auto output_db_hidden_shape = c->MakeShape({hidden_size_1});
    auto output_dx_shape = c->MakeShape({num_step, batch_size, input_size});
    auto output_dh_prev_shape = c->MakeShape({batch_size, hidden_size});
    c->set_output(0, output_dw_input_shape);
    c->set_output(1, output_dw_hidden_shape);
    c->set_output(2, output_db_input_shape);
    c->set_output(3, output_db_hidden_shape);
    c->set_output(4, output_dx_shape);
    c->set_output(5, output_dh_prev_shape);
    return Status::OK();
  });

REGISTER_OP("DynamicAUGRU")
  .Input("x: T")
  .Input("weight_input: T")
  .Input("weight_hidden: T")
  .Input("weight_att: T")
  .Input("bias_input: T")
  .Input("bias_hidden: T")
  .Input("seq_length: int32")
  .Input("init_h: T")
  .Output("y: T")
  .Output("output_h: T")
  .Output("update: T")
  .Output("update_att: T")
  .Output("reset: T")
  .Output("new: T")
  .Output("hidden_new: T")
  .Attr("T: {float16, float32}")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("activation: string")
  .Attr("gate_order: string")
  .Attr("reset_after: bool = true")
  .Attr("is_training: bool = true")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto weight_hidden_shape = c->input(2);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto hidden_size = c->Dim(weight_hidden_shape, 0);
    int32_t num_proj = 0;
    TF_RETURN_IF_ERROR(c->GetAttr("num_proj", &num_proj));

    ShapeHandle output_y_shape;
    if (num_proj == 0) {
      output_y_shape = c->MakeShape({num_step, batch_size, hidden_size});
    } else {
      auto num_proj_size = c->MakeDim(shape_inference::DimensionOrConstant(num_proj));
      DimensionHandle output_hidden_size;
      TF_RETURN_IF_ERROR(c->Min(num_proj_size, hidden_size, &output_hidden_size));
      output_y_shape = c->MakeShape({num_step, batch_size, output_hidden_size});
    }
    auto output_h_shape = c->MakeShape({num_step, batch_size, hidden_size});
    c->set_output(0, output_y_shape);
    c->set_output(1, output_h_shape);
    c->set_output(2, c->UnknownShape());
    c->set_output(3, c->UnknownShape());
    c->set_output(4, c->UnknownShape());
    c->set_output(5, c->UnknownShape());
    c->set_output(6, c->UnknownShape());
    return Status::OK();
  });

REGISTER_OP("DynamicAUGRUGrad")
  .Input("x: T")
  .Input("weight_input: T")
  .Input("weight_hidden: T")
  .Input("weight_att: T")
  .Input("y: T")
  .Input("init_h: T")
  .Input("h: T")
  .Input("dy: T")
  .Input("dh: T")
  .Input("update: T")
  .Input("update_att: T")
  .Input("reset: T")
  .Input("new: T")
  .Input("hidden_new: T")
  .Input("seq_length: int32")
  .Output("dw_input: T")
  .Output("dw_hidden: T")
  .Output("db_input: T")
  .Output("db_hidden: T")
  .Output("dx: T")
  .Output("dh_prev: T")
  .Output("dw_att: T")
  .Attr("T: {float16, float32}")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("gate_order: string")
  .Attr("reset_after: bool = true")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto weight_hidden_shape = c->input(2);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto input_size = c->Dim(input_shape, 2);
    auto hidden_size = c->Dim(weight_hidden_shape, 0);
    auto hidden_size_1 = c->Dim(weight_hidden_shape, 1);
    auto output_dw_input_shape = c->MakeShape({input_size, hidden_size_1});
    auto output_dw_hidden_shape = c->MakeShape({hidden_size, hidden_size_1});
    auto output_db_input_shape = c->MakeShape({hidden_size_1});
    auto output_db_hidden_shape = c->MakeShape({hidden_size_1});
    auto output_dx_shape = c->MakeShape({num_step, batch_size, input_size});
    auto output_dh_prev_shape = c->MakeShape({batch_size, hidden_size});
    auto output_dw_att_shape = c->MakeShape({num_step, batch_size});
    c->set_output(0, output_dw_input_shape);
    c->set_output(1, output_dw_hidden_shape);
    c->set_output(2, output_db_input_shape);
    c->set_output(3, output_db_hidden_shape);
    c->set_output(4, output_dx_shape);
    c->set_output(5, output_dh_prev_shape);
    c->set_output(6, output_dw_att_shape);
    return Status::OK();
  });

REGISTER_OP("DynamicRnn")
  .Input("x: T")
  .Input("w: T")
  .Input("b: T")
  .Input("seq_length: int32")
  .Input("init_h: T")
  .Input("init_c: T")
  .Output("y: T")
  .Output("output_h: T")
  .Output("output_c: T")
  .Output("i: T")
  .Output("j: T")
  .Output("f: T")
  .Output("o: T")
  .Output("tanhc: T")
  .Attr("T: {float16, float32}")
  .Attr("cell_type: string")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("use_peephole: bool = false")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("activation: string")
  .Attr("forget_bias: float = 0.0")
  .Attr("is_training: bool = true")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto input_size = c->Dim(input_shape, 2);
    auto w = c->input(1);
    auto hidden_size_total = c->Dim(w, 0);
    DimensionHandle hidden_size;
    TF_RETURN_IF_ERROR(c->Subtract(hidden_size_total, input_size, &hidden_size));
    int32_t num_proj = 0;
    TF_RETURN_IF_ERROR(c->GetAttr("num_proj", &num_proj));
    ShapeHandle output_y_shape;
    if (num_proj == 0) {
      output_y_shape = c->MakeShape({num_step, batch_size, hidden_size});
    } else {
      auto num_proj_size = c->MakeDim(shape_inference::DimensionOrConstant(num_proj));
      DimensionHandle output_hidden_size;
      TF_RETURN_IF_ERROR(c->Min(num_proj_size, hidden_size, &output_hidden_size));
      output_y_shape = c->MakeShape({num_step, batch_size, output_hidden_size});
    }
    auto output_h_shape = c->MakeShape({num_step, batch_size, hidden_size});
    auto output_c_shape = c->MakeShape({num_step, batch_size, hidden_size});

    c->set_output(0, output_y_shape);
    c->set_output(1, output_h_shape);
    c->set_output(2, output_c_shape);
    c->set_output(3, c->UnknownShape());
    c->set_output(4, c->UnknownShape());
    c->set_output(5, c->UnknownShape());
    c->set_output(6, c->UnknownShape());
    c->set_output(7, c->UnknownShape());
    return Status::OK();
  });

REGISTER_OP("DynamicRnnV2")
  .Input("x: T")
  .Input("w: T")
  .Input("b: T")
  .Input("init_h: T")
  .Input("init_c: T")
  .Output("y: T")
  .Output("output_h: T")
  .Output("output_c: T")
  .Output("i: T")
  .Output("j: T")
  .Output("f: T")
  .Output("o: T")
  .Output("tanhc: T")
  .Attr("T: {float16, float32}")
  .Attr("cell_type: string")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("use_peephole: bool = false")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("activation: string")
  .Attr("forget_bias: float = 0.0")
  .Attr("is_training: bool = true")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto input_size = c->Dim(input_shape, 2);
    auto w = c->input(1);
    auto hidden_size_total = c->Dim(w, 0);
    DimensionHandle hidden_size;
    TF_RETURN_IF_ERROR(c->Subtract(hidden_size_total, input_size, &hidden_size));
    int32_t num_proj = 0;
    TF_RETURN_IF_ERROR(c->GetAttr("num_proj", &num_proj));
    ShapeHandle output_y_shape;
    if (num_proj == 0) {
      output_y_shape = c->MakeShape({num_step, batch_size, hidden_size});
    } else {
      auto num_proj_size = c->MakeDim(shape_inference::DimensionOrConstant(num_proj));
      DimensionHandle output_hidden_size;
      TF_RETURN_IF_ERROR(c->Min(num_proj_size, hidden_size, &output_hidden_size));
      output_y_shape = c->MakeShape({num_step, batch_size, output_hidden_size});
    }
    auto output_h_shape = c->MakeShape({num_step, batch_size, hidden_size});
    auto output_c_shape = c->MakeShape({num_step, batch_size, hidden_size});

    c->set_output(0, output_y_shape);
    c->set_output(1, output_h_shape);
    c->set_output(2, output_c_shape);
    c->set_output(3, c->UnknownShape());
    c->set_output(4, c->UnknownShape());
    c->set_output(5, c->UnknownShape());
    c->set_output(6, c->UnknownShape());
    c->set_output(7, c->UnknownShape());
    return Status::OK();
  });

REGISTER_OP("DynamicRnnGrad")
  .Input("x: T")
  .Input("w: T")
  .Input("b: T")
  .Input("y: T")
  .Input("init_h: T")
  .Input("init_c: T")
  .Input("h: T")
  .Input("c: T")
  .Input("dy: T")
  .Input("dh: T")
  .Input("dc: T")
  .Input("i: T")
  .Input("j: T")
  .Input("f: T")
  .Input("o: T")
  .Input("tanhc: T")
  .Output("dw: T")
  .Output("db: T")
  .Output("dx: T")
  .Output("dh_prev: T")
  .Output("dc_prev: T")
  .Attr("T: {float16, float32}")
  .Attr("cell_type: string")
  .Attr("direction: string")
  .Attr("cell_depth: int = 1")
  .Attr("use_peephole: bool = false")
  .Attr("keep_prob: float = 1.0")
  .Attr("cell_clip: float = -1.0")
  .Attr("num_proj: int = 0")
  .Attr("time_major: bool = true")
  .Attr("forget_bias: float = 0.0")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    auto num_step = c->Dim(input_shape, 0);
    auto batch_size = c->Dim(input_shape, 1);
    auto input_size = c->Dim(input_shape, 2);
    auto w = c->input(1);
    auto hidden_size_total = c->Dim(w, 0);
    auto hidden_size_4 = c->Dim(w, 1);
    DimensionHandle hidden_size;
    TF_RETURN_IF_ERROR(c->Subtract(hidden_size_total, input_size, &hidden_size));

    auto output_dx_shape = c->MakeShape({num_step, batch_size, input_size});
    auto output_dw_shape = c->MakeShape({hidden_size_total, hidden_size_4});
    auto output_db_shape = c->MakeShape({hidden_size_4});
    auto output_dh_prev_shape = c->MakeShape({1, batch_size, hidden_size});
    auto output_dc_prev_shape = c->MakeShape({1, batch_size, hidden_size});
    c->set_output(0, output_dw_shape);
    c->set_output(1, output_db_shape);
    c->set_output(2, output_dx_shape);
    c->set_output(3, output_dh_prev_shape);
    c->set_output(4, output_dc_prev_shape);
    return Status::OK();
  });

REGISTER_OP("EmbeddingHashTableLookupOrInsert")
  .Input("table_handle: int64")
  .Input("keys:int64")
  .Output("values: float")
  .Attr("bucket_size:int")
  .Attr("embedding_dim:int")
  .Attr("filter_mode:string='no_filter'")
  .Attr("filter_freq:int=0")
  .Attr("default_key_or_value:bool = false")
  .Attr("default_key: int = 0")
  .Attr("default_value: float = 0.0")
  .Attr("filter_key_flag: bool = false")
  .Attr("filter_key: int = -1")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    int64 num = 0;
    c->GetAttr("embedding_dim", &num);
    auto key_num = c->input(1);
    int64_t nsample = InferenceContext::Value(c->Dim(key_num, 0));
    c->set_output(0, c->MakeShape({c->MakeDim(nsample), c->MakeDim(num)}));
    return Status::OK();
  });

REGISTER_OP("LRUCacheV2")
  .Input("index_list: T")
  .Input("data: Ref(dtype)")
  .Input("cache: Ref(dtype)")
  .Input("tag: Ref(T)")
  .Input("is_last_call: T")
  .Output("data1:Ref(dtype)")
  .Output("cache1: Ref(dtype)")
  .Output("tag1: Ref(dtype)")
  .Output("index_offset_list: T")
  .Output("not_in_cache_index_list: T")
  .Output("not_in_cache_number: T")
  .Attr("T: {int32, int64}")
  .Attr("dtype: {float32}")
  .Attr("pre_route_count: int")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(1));
    c->set_output(1, c->input(2));
    c->set_output(2, c->input(3));
    c->set_output(3, c->input(0));
    c->set_output(4, c->input(0));
    c->set_output(5, c->MakeShape({1}));
    return Status::OK();
  });

REGISTER_OP("Centralization")
  .Input("x: T")
  .Output("y: T")
  .Attr("T: {float16, float32}")
  .Attr("axes: list(int)")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("PRelu")
  .Input("x: T")
  .Input("weight: T")
  .Output("y: T")
  .Attr("T: {float16, float32}")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("DropOutDoMaskV3")
  .Input("x: T")
  .Input("mask: uint8")
  .Input("keep_prob: T")
  .Output("y: T")
  .Attr("T: {float16, float32}")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("PReluGrad")
  .Input("grads: T")
  .Input("features: T")
  .Input("weights: T")
  .Output("dx: T")
  .Output("da: T")
  .Attr("T: {float16, float32}")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(2));
    return Status::OK();
  });

REGISTER_OP("NonZero")
  .Input("x:T")
  .Output("y:output_type")
  .Attr("transpose:bool = false")
  .Attr("T:numbertype")
  .Attr("output_type:{int32, int64} = DT_INT64")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto rank = InferenceContext::Rank(c->input(0));
    c->set_output(0, c->MakeShape({rank, -1}));
    return Status::OK();
  });

REGISTER_OP("NonZeroWithValue")
  .Input("x:T")
  .Output("value:T")
  .Output("index:output_type")
  .Output("count:output_type")
  .Attr("transpose:bool = false")
  .Attr("T:numbertype")
  .Attr("output_type:{int32, int64} = DT_INT32")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext *c) {
    auto input_shape = c->input(0);
    int64_t dim1 = InferenceContext::Value(c->Dim(input_shape, 0));
    int64_t dim2 = InferenceContext::Value(c->Dim(input_shape, 1));
    int64_t value_num = dim1 * dim2;
    int64_t index_dim = 2 * dim1 * dim2;
    int64_t count_dim = 1;

    c->set_output(0, c->MakeShape({c->MakeDim(value_num)}));
    c->set_output(1, c->MakeShape({c->MakeDim(index_dim)}));
    c->set_output(2, c->MakeShape({c->MakeDim(count_dim)}));
    return Status::OK();
  });

REGISTER_OP("FusedLayerNorm")
  .Input("x: T")
  .Input("gamma: T")
  .Input("beta: T")
  .Output("y: T")
  .Output("mean: T")
  .Output("variance: T")
  .Attr("T: {float16, float32}")
  .Attr("begin_norm_axis: int = 0")
  .Attr("begin_params_axis: int = 0")
  .Attr("epsilon: float = 0.0000001")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    int32_t real_dim_num = InferenceContext::Rank(c->input(0));
    int32_t begin_norm_axis = 0;
    TF_RETURN_IF_ERROR(c->GetAttr("begin_norm_axis", &begin_norm_axis));
    if (begin_norm_axis < 0) {
      begin_norm_axis += real_dim_num;
    }
    if (begin_norm_axis < 0 || begin_norm_axis >= real_dim_num) {
      return errors::InvalidArgument("begin_norm_axis is invalid");
    }
    ShapeHandle input_shape_handle;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), real_dim_num, &input_shape_handle));
    ShapeHandle out_shape_handle;
    for (int32_t i = 0; i < real_dim_num; ++i) {
      DimensionHandle tmp_dim_handle = c->Dim(input_shape_handle, i);
      if (i >= begin_norm_axis) {
        tmp_dim_handle = c->MakeDim(1);
        TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape_handle, i, tmp_dim_handle, &out_shape_handle));
      }
    }
    c->set_output(0, c->input(0));
    c->set_output(1, out_shape_handle);
    c->set_output(2, out_shape_handle);
    return Status::OK();
  });

REGISTER_OP("FusedLayerNormGrad")
  .Input("dy: T")
  .Input("x: T")
  .Input("variance: T")
  .Input("mean: T")
  .Input("gamma: T")
  .Output("pd_x: T")
  .Output("pd_gamma: T")
  .Output("pd_beta: T")
  .Attr("T: {float16, float32}")
  .SetIsStateful()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(4));
    c->set_output(2, c->input(4));
    return Status::OK();
  });

REGISTER_OP("GetShape")
  .Input("x: T")
  .Output("y: int32")
  .Attr("N: int = 1")
  .Attr("T: {float16, float32, uint8}")
  .SetShapeFn([](InferenceContext* c) {
    int64_t sumSize = 0;
    for (int32_t i = 0; i < c->num_inputs(); i++) {
      sumSize += InferenceContext::Rank(c->input(i));
    }
    c->set_output(0, c->MakeShape({c->MakeDim(sumSize)}));
    return Status::OK();
  });

REGISTER_OP("ProdEnvMatA")
  .Input("coord: T")
  .Input("type:int32")
  .Input("natoms:int32")
  .Input("box: T")
  .Input("mesh:int32")
  .Input("davg: T")
  .Input("dstd: T")
  .Output("descrpt: T")
  .Output("descrpt_deriv: T")
  .Output("rij: T")
  .Output("nlist:int32")
  .Attr("T: {float16, float32}")
  .Attr("rcut_a: float = 0.0")
  .Attr("rcut_r: float = 0.0")
  .Attr("rcut_r_smth: float = 0.0")
  .Attr("sel_a: list(int)")
  .Attr("sel_r: list(int)")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    auto coord_shape = c->input(0);
    int64_t nsample = InferenceContext::Value(c->Dim(coord_shape, 0));
    int64_t nloc = 12288;
    int64_t nnei = 0;
    std::vector<int32_t> sel_a;
    TF_RETURN_IF_ERROR(c->GetAttr("sel_a", &sel_a));
    for (size_t i = 0; i < sel_a.size(); ++i) {
      nnei = nnei + sel_a[i];
    }
    int64_t des = nloc * nnei * 4;
    int64_t des_a = des * 3;
    int64_t rij = nloc * nnei * 3;
    int64_t nlist = nloc * nnei;
    c->set_output(0, c->MakeShape({c->MakeDim(nsample), c->MakeDim(des)}));
    c->set_output(1, c->MakeShape({c->MakeDim(nsample), c->MakeDim(des_a)}));
    c->set_output(2, c->MakeShape({c->MakeDim(nsample), c->MakeDim(rij)}));
    c->set_output(3, c->MakeShape({c->MakeDim(nsample), c->MakeDim(nlist)}));
    return Status::OK();
  });

REGISTER_OP("ProdVirialSeA")
  .Input("net_deriv:T")
  .Input("in_deriv:T")
  .Input("rij:T")
  .Input("nlist:int32")
  .Input("natoms:int32")
  .Output("virial:T")
  .Output("atom_virial:T")
  .Attr("n_a_sel:int = 0")
  .Attr("n_r_sel:int = 0")
  .Attr("T: {float32, float64}")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    auto input_shape = c->input(0);
    auto nframes = c->Dim(input_shape, 0);
    ShapeHandle virial_shape = c->MakeShape({nframes, 9});
    c->set_output(0, virial_shape);
    ShapeHandle atom_virial_shape = c->MakeShape({nframes, 254952});
    c->set_output(1, atom_virial_shape);
    return Status::OK();
  });

REGISTER_OP("ProdForceSeA")
  .Input("net_deriv:T")
  .Input("in_deriv:T")
  .Input("nlist:int32")
  .Input("natoms:int32")
  .Output("force:T")
  .Attr("n_a_sel:int = 0")
  .Attr("n_r_sel:int = 0")
  .Attr("T: {float32}")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    auto input_shape = c->input(0);
    auto nframes = c->Dim(input_shape, 0);
    ShapeHandle force_shape = c->MakeShape({nframes, 84984});
    c->set_output(0, force_shape);
    return Status::OK();
  });

REGISTER_OP("TabulateFusionSeA")
  .Input("table:T")
  .Input("table_info:T")
  .Input("em_x:T")
  .Input("em:T")
  .Output("descriptor:T")
  .Attr("last_layer_size:int")
  .Attr("T: {float16, float32, float64}")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    auto input_shape = c->input(3);
    auto nloc = c->Dim(input_shape, 0);

    int32_t last_layer_size;
    TF_RETURN_IF_ERROR(c->GetAttr("last_layer_size", &last_layer_size));
    ShapeHandle out_shape = c->MakeShape({nloc, 4, last_layer_size});
    c->set_output(0, out_shape);
    return Status::OK();
  });

REGISTER_OP("TabulateFusionSeAGrad")
  .Input("table:T")
  .Input("table_info:T")
  .Input("em_x:T")
  .Input("em:T")
  .Input("dy:T")
  .Input("descriptor:T")
  .Output("dy_dem_x:T")
  .Output("dy_dem:T")
  .Attr("T: {float16, float32, float64}")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    c->set_output(0, c->input(2));
    c->set_output(1, c->input(3));
    return Status::OK();
  });

REGISTER_OP("TabulateFusion")
  .Input("table:T")
  .Input("table_info:T")
  .Input("em_x:T")
  .Input("em:T")
  .Output("descriptor:T")
  .Attr("last_layer_size:int")
  .Attr("T: {float16, float32, float64}")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    auto input_shape = c->input(3);
    auto nloc = c->Dim(input_shape, 0);

    int32_t last_layer_size;
    TF_RETURN_IF_ERROR(c->GetAttr("last_layer_size", &last_layer_size));
    ShapeHandle out_shape = c->MakeShape({nloc, 4, last_layer_size});
    c->set_output(0, out_shape);
    return Status::OK();
  });

REGISTER_OP("TabulateFusionGrad")
  .Input("table:T")
  .Input("table_info:T")
  .Input("em_x:T")
  .Input("em:T")
  .Input("dy:T")
  .Input("descriptor:T")
  .Output("dy_dem_x:T")
  .Output("dy_dem:T")
  .Attr("T: {float16, float32, float64}")
  .SetIsStateful()
  .SetShapeFn([](InferenceContext* c) {
    c->set_output(0, c->input(2));
    c->set_output(1, c->input(3));
    return Status::OK();
  });

REGISTER_OP("InitEmbeddingHashTable")
  .Input("table_handle: int64")
  .Input("sampled_values: float")
  .Attr("bucket_size : int")
  .Attr("embedding_dim : int")
  .Attr("initializer_mode : string='random'")
  .Attr("constant_value : float=0.0")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingHashTableEvict")
  .Input("table_handle: int64")
  .Input("keys: int64")
  .Attr("table_cap: int")
  .Attr("embedding_dim: int")
  .Attr("init_mode: string='constant'")
  .Attr("const_val: float=0.0")
  .SetIsStateful()
  .SetShapeFn(shape_inference::NoOutputs);
}  // namespace
}  // namespace tensorflow
