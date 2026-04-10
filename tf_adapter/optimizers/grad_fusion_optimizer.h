/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_GRADIENT_FUSION_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_GRADIENT_FUSION_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tf_adapter/common/compat_tf1_tf2.h"

namespace tensorflow {
namespace grappler {
class GradFusionOptimizer : public CustomGraphOptimizer {
 public:
  GradFusionOptimizer() {}

  ~GradFusionOptimizer() override = default;

  string name() const override {
    return "GradFusionOptimizer";
  }

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer *config) override {
    (void) config;
    return Status::OK();
  }

  bool UsesFunctionLibrary() const override {
    return false;
  }

  Status Optimize(Cluster *cluster, const GrapplerItem &item, GraphDef *optimizedGraph) override;

  VOID_FUNCTION_ONLY_TF1(Feedback(Cluster *cluster, const GrapplerItem &item, const GraphDef &optimizedGraph,
                                  double result) override)
};
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_GRADIENT_FUSION_OPTIMIZER_H_
