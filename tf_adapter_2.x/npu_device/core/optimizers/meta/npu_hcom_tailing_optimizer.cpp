/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_device.h"
#include "optimizers/npu_optimizer_manager.h"

const static std::string kHcomAllReduce = "HcomAllReduce";
const static std::string kNpuLossScaleAttr = "_npu_loss_scale";
const static std::string kNpuAllocFloatStatusOp = "NpuAllocFloatStatus";
const static std::string kEnable = "1";

/**
 * @brief: tailing optimize
 * @param context: tfe context
 * @param graph: tensorflow graph
 * @param changed: if changed or not
 */
namespace {
tensorflow::Status TailingOptimizeInner(tensorflow::FunctionLibraryDefinition *lib_def, tensorflow::Graph *graph,
                                        bool &changed) {
  for (tensorflow::Node *node : graph->op_nodes()) {
    for (auto &attr : node->attrs()) {
      if (attr.second.has_func()) {
        std::string func_name = attr.second.func().name();
        const tensorflow::FunctionDef *fdef = lib_def->Find(func_name);
        std::unique_ptr<tensorflow::FunctionBody> fbody;
        NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));
        bool optimized = false;
        NPU_REQUIRES_OK(TailingOptimizeInner(lib_def, fbody->graph, optimized));
        if (optimized) {
          tensorflow::FunctionDef optimized_fdef;
          auto lookup = [&fdef](const tensorflow::Node *node) -> absl::optional<std::string> {
            for (const auto &control_ret : fdef->control_ret()) {
              if (control_ret.second == node->name()) {
                return absl::make_optional(node->name());
              }
            }
            return absl::nullopt;
          };
          NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*fbody->graph, func_name, lookup, &optimized_fdef));
          NPU_REQUIRES_OK(lib_def->RemoveFunction(func_name));
          NPU_REQUIRES_OK(lib_def->AddFunctionDef(optimized_fdef));
        }
      }
    }
    if ((node->type_string() == kNpuAllocFloatStatusOp) && (node->attrs().Find(kNpuLossScaleAttr) != nullptr)) {
      std::unordered_set<const tensorflow::Edge *> edges_to_remove;
      tensorflow::Node *last_allreduce = nullptr;
      for (auto in_edge : node->in_edges()) {
        if (in_edge->IsControlEdge()) {
          if (last_allreduce == nullptr) {
            if (in_edge->src()->type_string() == kHcomAllReduce) {
              last_allreduce = in_edge->src();
            }
          }
          (void)edges_to_remove.insert(in_edge);
        }
      }
      if (last_allreduce == nullptr || edges_to_remove.empty()) {
        continue;
      }

      tensorflow::Node *float_status_allreduce = nullptr;
      for (auto out_edge : node->out_edges()) {
        if (out_edge->dst()->type_string() == kHcomAllReduce &&
            out_edge->dst()->attrs().Find(kNpuLossScaleAttr) != nullptr) {
          float_status_allreduce = out_edge->dst();
          break;
        }
      }
      if (float_status_allreduce == nullptr) {
        continue;
      }

      std::unordered_set<tensorflow::Node *> grads;
      tensorflow::Node *previous_allreduce = last_allreduce;
      while (previous_allreduce != nullptr) {
        const tensorflow::EdgeSet &in_edges = previous_allreduce->in_edges();
        previous_allreduce = nullptr;
        for (auto in_edge : in_edges) {
          if (!in_edge->IsControlEdge()) {
            (void)grads.insert(in_edge->src());
          } else if (in_edge->src()->type_string() == kHcomAllReduce) {
            previous_allreduce = in_edge->src();
          }
        }
      }

      if (!grads.empty()) {
        for (auto edge : edges_to_remove) {
          graph->RemoveEdge(edge);
        }
        for (auto grad : grads) {
          (void)graph->AddControlEdge(grad, node);
        }
        (void)graph->AddControlEdge(float_status_allreduce, last_allreduce);
        changed = true;
      }
    }
  }

  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
tensorflow::Status HcomTailingOptimize(TFE_Context *context, tensorflow::Graph *graph,
                                       std::map<std::string, std::string> options) {
  if (options[ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION] != kEnable) {
    return tensorflow::Status::OK();
  }
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  bool unused = false;
  return TailingOptimizeInner(lib_def, graph, unused);
}

NPU_REGISTER_META_OPTIMIZER(1, "HcomTailingOptimizer", HcomTailingOptimize);
}  // namespace npu
