/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_CONTRIB_OFFLINE_TRAIN_UTIL_INFERSHAPE_H_
#define TENSORFLOW_CONTRIB_OFFLINE_TRAIN_UTIL_INFERSHAPE_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
const std::string KEY_SHAPE = "geop_shape";

class InferShapeUtil {
 public:
  static Status InferShape(const std::vector<Tensor> &vecTensor, const FunctionLibraryDefinition *flib_def,
                           const FunctionDef *func_def, Graph *graph);

  static Status GetSubGraphFromFunctionDef(const FunctionLibraryDefinition &flib_def, const FunctionDef &func_def,
                                           Graph *graph);

  static int64 GetCurrentTimestap();
  static bool IsInitializedGraph(const Node *node);

  static const int INFER_SHAPE_FIRST_TIME = 0;
  static const int INFER_SHAPE_OTHER_TIME = 1;

 private:
  static Status setArgShapeFromTensorShape(const std::vector<Tensor> vecTensor, const Graph *graph, const OpDef &sig,
                                           ShapeRefiner &shapeRef);

  static Status getInputShapesOfNode(const ShapeRefiner &shapeRef, const Node *pNode,
                                     std::vector<tensorflow::shape_inference::ShapeHandle> &inputShapeVec);

  static void setShapeOfEnterOP(const ShapeRefiner &shapeRef, const Node *pNode);

  static void setShapeOfMergeOP(const ShapeRefiner &shapeRef, const Node *pNode);

  static void inferShapeOfGraph(const Graph *graph, ShapeRefiner &shapeRef, int iTime);

  static Status addShapeToAttr(ShapeRefiner &shapeRef, Node *pNode);
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_OFFLINE_TRAIN_UTIL_INFERSHAPE_H_
