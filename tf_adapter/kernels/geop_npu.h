/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_KERNELS_GEOP_NPU_H_
#define TENSORFLOW_KERNELS_GEOP_NPU_H_

#include <unordered_map>
#include <atomic>
#include <functional>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

#include "ge/ge_api.h"
#include "ge/ge_api_wrapper.h"
#include "ge_common/ge_api_types.h"

namespace tensorflow {
using SessionId = uint64_t;
using AoeStatus = int32_t;
// aoe mode
using AoeInitializeFunc = AoeStatus (*)(const std::map<ge::AscendString, ge::AscendString> &);
using AoeFinalizeFunc = AoeStatus (*)();
using AoeCreateSessionFunc = AoeStatus (*)(SessionId &);
using AoeDestroySessionFunc = AoeStatus (*)(SessionId);
using AoeSetGeSessionFunc = AoeStatus (*)(SessionId, ge::Session*);
using AoeSetDependGraphFunc = AoeStatus (*)(SessionId, const std::vector<ge::Graph>&);
using AoeSetDependGraphsInputsFunc = AoeStatus (*)(SessionId, const std::vector<std::vector<ge::Tensor>> &);
using AoeSetTuningGraphInputFunc = AoeStatus (*)(SessionId, const std::vector<ge::Tensor> &);
using AoeSetTuningGraphFunc = AoeStatus (*)(SessionId, const ge::Graph &);
using AoeTuningGraphFunc = AoeStatus (*)(SessionId, const std::map<ge::AscendString, ge::AscendString> &);

enum GraphStatus {
  Init,
  CompileDone,
  Compiling
};

struct GraphHandler {
  GraphStatus status = Init;
  mutex graph_mu;
  condition_variable cv;
  int32_t graph_run_num = 0;
  ge::ComputeGraphPtr graph;
};

class GeOp : public AsyncOpKernel {
public:
  explicit GeOp(OpKernelConstruction *ctx);
  ~GeOp() override;
  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override;

  enum class FastValue { kfast = 0, kfast1 };
  struct AccelerateInfo {
    FastValue fast_value_;
    std::string fast_mode_;
    float fast_ratio_;
    std::string origin_precision_mode_v1;
    std::string origin_precision_mode_v2;
    bool is_inited_{false};            // mark has been parsered from option or not
    bool is_recovered_{false};         // mark has been triggered recover precision mode or not
    uint32_t iteration_per_loop_{0U};  // iteration_per_loop has some effect to match fast_ratio_
    Status JudgeNeedRecompile(bool &need_recompile);

   private:
    Status TriggeredByStep(bool &is_triggered);
    Status TriggeredByLoss(bool &is_triggered);
  };

 private:
  void Initialize(OpKernelConstruction *ctx);
  void Finalize();

  // global environment Initialize/Finalize, only invoke once for each process
  Status GlobalInitialize(OpKernelConstruction *ctx);
  void GlobalFinalize();

  // Build GraphDef from FunctionDef.
  Status BuildGraphDef(FunctionLibraryDefinition &flib_def, const std::vector<Tensor> &input_vec,
                       GraphDef &graph_def, bool &is_initialize, bool &is_allreduce);
  Status SeparateGraphDef(GraphDef &ori_graph_def,
                          std::vector<ge::AscendString> &partition_graph,
                          std::map<ge::AscendString, ge::AscendString> &const_value_map);
  // Analyze sting input data
  Status AnalyzeStringInput(ge::Tensor &input, const std::vector<std::string> &string_vector) const;

  // prepare input tensor
  Status BuildInputTensorInfo(OpKernelContext *const ctx,
                              std::vector<Tensor> &input_vec,
                              std::vector<std::string> &input_shapes,
                              std::vector<ge::Tensor> &inputs);
  // prepare output tensor
  Status BuildOutTensorInfo(OpKernelContext *ctx);

  Status ParserGraph(OpKernelContext *ctx, const std::vector<Tensor> &input_vec);
  Status AddGraph(OpKernelContext *ctx, const uint32_t &graph_id);
  Status CompileGraph(OpKernelContext *ctx, const std::vector<Tensor> &input_vec,
                      const std::vector<ge::Tensor> &inputs,
                      const uint32_t &graph_id);
  Status BuildGraph(const uint32_t &graph_id, const std::vector<ge::Tensor> &inputs);
  Status RunGraph(OpKernelContext *ctx, const uint32_t &graph_id,
                  const std::shared_ptr<std::vector<ge::Tensor>> &inputs,
                  DoneCallback done);
  Status CompileAndRunGraph(OpKernelContext *ctx,
                            const std::vector<Tensor> &input_vec,
                            const std::shared_ptr<std::vector<ge::Tensor>> &inputs,
                            const std::vector<std::string> &input_shapes,
                            DoneCallback done);
  bool IsLazyCompile();
  // create input and output desc for NodeDef
  Status GenerateDesc(Node *&node);

  // parse onnx model in tensorflow node
  Status ParseOnnxGraphOpAttr(Node *&node) const;

  Status DomiFormatFromString(std::string format, int32_t &domi_format) const;

  Status GraphInputConvertToConst(OpKernelContext *ctx);

  Status GraphCheckInputEqualConstOp(Tensor &tensor, int32_t index, bool &is_equal);

  void AddNodeAttrs(Node *node, bool &is_initialize);

  bool IsGraphNeedRebuild(const uint32_t cache_graph_id);
  Status DoAccelerateTrain();
  Status NeedRecompileWhenAccelerateTrainOn(bool &need_recompile);
  bool IsAccelerateTrainOn();
  Status ParserAccelerateTrain(const std::string &accelerate_train_mode);
  Status CheckAndSetAccelarateMode(const std::string &mode_value);
  Status CheckAndSetAccelarateRatio(const std::string &mode_value, const std::string &ratio_value);
  Status CheckAndModifyPrecisionMode();
  Status RecoverPrecisionMode();
  bool IncrementGraphIdCount(uint32_t &graph_id);

  bool DecrementGraphIdCount(uint32_t &graph_id);

  void ClearGraphIdCount();

  void GetExecGraphId(uint32_t &cache_graph_id,
                      const std::vector<std::string> &input_shapes);

  void GetMsTuneConfig(std::map<std::string, std::string> init_options);

  void SetShapesToOutputDesc(const std::vector<std::string> &input_shapes,
                             const int &index, AttrValue &attr_shape_value) const;

  void BuildShapeNodeAndCacheArgNodes(Graph &graph);

  Status ChangeInputsShapeDesc();

  void AnalyzeInputDesc(bool need_collect_shapes, void *tensor_ptr, ge::Tensor &input, ge::DataType type,
                        std::vector<std::string> &input_shapes) const;

  int RunTuning(std::vector<Tensor> &input_vec, std::vector<ge::Tensor> &inputs, const OpKernelContext *const ctx);

  int ExecuteAoeTuning(ge::Graph &ge_graph, bool is_allreduce, std::vector<ge::Tensor> &inputs);

  std::string BuildSubGraph(FunctionLibraryDefinition *flib_def, const std::string &graph);

  void SetDynamicInput();
  Status SetGraphOptions(OpKernelContext *ctx);
  void ProcessDpOpFuncDef(const Node &node) const;

  void BuildQueueDataAndGetNextFromQueue(Graph &graph, const Node &getnext_node,
                                         const std::string &channel_name) const;

  void HandleDpOpAndGetNextNodes(Graph &graph);

  bool IsDynamicGetNext(const Node *node);

  void ChangeChannelNameAttr(NodeDef &node_def) const;
  void InitGraphShape(OpKernelContext *const ctx);
  bool IsDynamicConfig();

  PartialTensorShape MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b) const;
  PartialTensorShape MakeAdaptiveShape(const PartialTensorShape &a, const PartialTensorShape &b) const;

  bool MaybeUpdateShape(OpKernelContext *const ctx);
  PartialTensorShape MakeUnknownShape(const int32_t &size) const;
  Status ProcessForDiffNodeTypes(Graph &graph, bool &is_initialize, bool &is_allreduce);

  void ProcessGetNextNode(const Node *node);

  void UpdateInputsShapeDesc(Graph &graph);

  Status DoGraphParser(ge::ComputeGraphPtr &compute_graph, FunctionLibraryDefinition *flib_def,
                       GraphDef &ori_graph_def);

  Status CreateGeSession();
  void InitAoeFlag();
  static const std::string INPUT_DESC;
  static const std::string OUTPUT_DESC;
  static const std::string SERIALIZE_FORMAT;
  static const std::string SERIALIZE_DATATYPE;
  static const std::string SERIALIZE_SHAPE;
  static const std::string SubGraph;

  static mutex mu_;
  static bool tuned_initialize_flag_;

  bool init_flag_;
  bool sess_init_flag_;
  bool graph_id_init_flag_;
  bool is_input_convert_;

  NameAttrList function_;
  std::string data_format_;
  uint32_t graph_id_;
  bool is_initialized_graph_;
  bool is_empty_graph_;
  bool need_iteration_;
  std::string tf_session_;
  ge::Session *ge_session_;
  std::string job_type_;
  std::string mix_compile_mode_;
  std::string accelerate_train_mode_;
  std::map<std::vector<std::string>, uint32_t> cache_graphs_;
  std::vector<std::pair<std::vector<std::string>, uint32_t>> graph_counts_;
  std::map<std::string, std::string> sess_options_;
  std::map<std::string, std::string> init_options_;
  static std::unordered_map<std::string, uint32_t> session_and_graph_id_map_;
  uint32_t iteration_per_loop_;
  bool is_host_graph_;
  std::map<std::string, std::string> graph_options_;
  std::map<int, TensorShape> outputs_shape_;
  std::string is_train_graph_;
  void *handle_;
  std::vector<Node*> dynamic_shape_nodes_;
  std::string dynamic_input_;
  std::string compile_dynamic_mode_;
  std::string shape_generalization_mode_{"STRICT"};
  uint32_t graph_max_parallel_model_num_{1U};
  std::string dynamic_graph_execute_mode_;
  std::string data_inputs_shape_range_;
  std::string getnext_inputs_shape_range_;
  std::string is_dynamic_getnext_;
  std::string placeholder_index_;
  std::atomic_flag tuned_flag_;
  std::vector<std::pair<Tensor, int32_t>> remove_index_;
  std::string is_var_init_graph_;
  std::string recompute_mode_;
  std::vector<absl::optional<PartialTensorShape>> input_shapes_vec_;
  std::string jit_compile_;
  bool is_dynamic_input_;
  std::map<std::string, bool> is_getnext_dynamic_shape_;
  SessionId session_id_;
  bool is_aoe_{false};
  bool need_recover_precision_mode_{false};
  AoeInitializeFunc aoe_initialize_;
  AoeFinalizeFunc aoe_finalize_;
  AoeCreateSessionFunc aoe_create_session_;
  AoeDestroySessionFunc aoe_destroy_session_;
  AoeSetGeSessionFunc aoe_set_gesession_;
  AoeSetDependGraphFunc aoe_set_dependgraphs_;
  AoeSetTuningGraphFunc aoe_set_tuninggraph_;
  AoeTuningGraphFunc aoe_tuning_graph_;
  AoeSetDependGraphsInputsFunc aoe_set_depend_graphs_inputs_;
  AoeSetTuningGraphInputFunc aoe_set_tuning_graph_input_;
  // accelerate train
  AccelerateInfo accelerate_info_;
  GraphHandler graph_handler_;
  bool need_compile_graph_first_;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_GEOP_NPU_H_
