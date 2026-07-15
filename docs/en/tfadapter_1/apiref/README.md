# TF Adapter 1.x API

- [TF Adapter Interface Overview](Tfadapter_interface_overview.md)
- [Session Configuration<a name="sub_menu"></a>](./session_config/README.md)
- [npu_bridge.estimator.npu.npu_config](./npu_config/npu_config.md)
  - [NPURunConfig Constructor<a name="sub_menu"></a>](./npu_config/npurunconfig_constructor/README.md)
  - [Supported RunConfig Parameters](./npu_config/runconfig_params_support_info.md)
  - [ProfilingConfig Constructor](./npu_config/profilingconfig_constructor.md)
  - [DumpConfig Constructor](./npu_config/dumpconfig_constructor.md)
  - [MemoryConfig Constructor](./npu_config/memoryconfig_constructor.md)
  - [ExperimentalConfig Constructor](./npu_config/experimentalconfig_constructor.md)

- [npu_bridge.estimator.npu.npu_estimator](./npu_estimator/npu_estimator.md)
  - [NPUEstimator Constructor](./npu_estimator/NPUEstimator_constructor.md)
  - [NPUEstimatorSpec Constructor](./npu_estimator/NPUEstimatorSpec_constructor.md)

- [npu_bridge.estimator.npu.npu_strategy](./npu_strategy/npu_strategy.md)
  - [NPUStrategy Constructor](./npu_strategy/NPUStrategy_constructor.md)

- [npu_bridge.estimator.npu.npu_hook](./npu_hook/npu_hook.md)
  - [NPUCheckpointSaverHook Constructor](./npu_hook/NPUCheckpointSaverHook_constructor.md)
  - [NPUOutputTensorHook Constructor](./npu_hook/NPUOutputTensorHook_constructor.md)
  - [TellMeStepOrLossHook Constructor](./npu_hook/TellMeStepOrLossHook_constructor.md)

- [npu_bridge.estimator.npu.npu_optimizer](./npu_optimizer/npu_optimizer.md)
  - [NPUDistributedOptimizer Constructor](./npu_optimizer/NPUDistributedOptimizer_constructor.md)
  - [NPUOptimizer Constructor](./npu_optimizer/NPUOptimizer_constructor.md)
  - [KerasDistributeOptimizer Constructor](./npu_optimizer/KerasDistributeOptimizer_constructor.md)
  - [npu_distributed_optimizer_wrapper](./npu_optimizer/npu_distributed_optimizer_wrapper.md)
  - [npu_allreduce](./npu_optimizer/npu_allreduce.md)

- [npu_bridge.estimator.npu.npu_callbacks](./npu_callbacks/npu_callbacks.md)
  - [NPUBroadcastGlobalVariablesCallback Constructor](./npu_callbacks/NPUBroadcastGlobalVariablesCallback_constructor.md)

- [npu_bridge.estimator.npu.npu_loss_scale_optimizer](./npu_loss_scale_optimizer/npu_loss_scale_optimizer.md)
  - [NPULossScaleOptimizer Constructor](./npu_loss_scale_optimizer/NPULossScaleOptimizer_constructor.md)

- [npu_bridge.estimator.npu.npu_loss_scale_manager](./npu_loss_scale_manager/npu_loss_scale_manager.md)
  - [FixedLossScaleManager Constructor](./npu_loss_scale_manager/FixedLossScaleManager_constructor.md)
  - [ExponentialUpdateLossScaleManager Constructor](./npu_loss_scale_manager/ExponentialUpdateLossScaleManager_constructor.md)

- [npu_bridge.estimator.npu_ops](./npu_ops/npu_ops.md)
  - [dropout](./npu_ops/dropout.md)
  - [LARSV2](./npu_ops/LARSV2.md)
  - [initialize_system](./npu_ops/initialize_system.md)
  - [shutdown_system](./npu_ops/shutdown_system.md)
  - [npu_onnx_graph_op](./npu_ops/npu_onnx_graph_op.md)

- [npu_bridge.estimator.npu.npu_rnn](./npu_rnn/npu-npu_rnn.md)
  - [npu_dynamic_rnn](./npu_rnn/npu_dynamic_rnn.md)

- [npu_bridge.estimator.npu.npu_dynamic_rnn](./npu_dynamic_rnn/npu_dynamic_rnn.md)
  - [DynamicRNN Constructor](./npu_dynamic_rnn/DynamicRNN_constructor.md)
  - [DynamicGRUV2 Constructor](./npu_dynamic_rnn/DynamicGRUV2_constructor.md)

- [npu_bridge.estimator.npu.npu_scope](./npu_scope/npu-npu_scope.md)
  - [without_npu_compile_scope](./npu_scope/without_npu_compile_scope.md)
  - [keep_dtype_scope](./npu_scope/keep_dtype_scope.md)
  - [npu_weight_prefetch_scope](./npu_scope/npu_weight_prefetch_scope.md)
  - [subgraph_multi_dims_scope](./npu_scope/subgraph_multi_dims_scope.md)
  - [disable_autofuse](./npu_scope/disable_autofuse.md)

- [npu_bridge.estimator.npu.util](./npu_util/npu-util.md)
  - [set_iteration_per_loop](./npu_util/set_iteration_per_loop.md)
  - [create_iteration_per_loop_var](./npu_util/create_iteration_per_loop_var.md)
  - [load_iteration_per_loop_var](./npu_util/load_iteration_per_loop_var.md)
  - [set_graph_exec_config](./npu_util/set_graph_exec_config.md)
  - [keep_tensors_dtypes](./npu_util/keep_tensors_dtypes.md)
  - [set_op_input_tensor_multi_dims](./npu_util/set_op_input_tensor_multi_dims.md)

- [npu_bridge.estimator.npu.keras_to_npu](./keras_to_npu/keras_to_npu.md)
  - [model_to_npu_estimator](./keras_to_npu/model_to_npu_estimator.md)

- [npu_bridge.estimator.npu.npu_plugin](./npu_plugin/npu_plugin.md)
  - [set_device_sat_mode](./npu_plugin/set_device_sat_mode.md)

- [npu_bridge.scoped_graph_manager.scoped_graph_manager](./scoped_graph_manager/scoped_graph_manager.md)
  - [ScopedGraphManager](./scoped_graph_manager/ScopedGraphManager.md)

- [npu_bridge.profiler.profiler](./profiler/profiler.md)
  - [Profiler Constructor](./profiler/Profiler_constructor.md)

- [npu_bridge.hccl.hccl_ops](./hccl_ops/hccl_ops.md)
  - [Description](./hccl_ops/introduction.md)
  - [allreduce](./hccl_ops/allreduce.md)
  - [allgather](./hccl_ops/allgather.md)
  - [broadcast](./hccl_ops/broadcast.md)
  - [reduce_scatter](./hccl_ops/reduce_scatter.md)
  - [reduce](./hccl_ops/reduce.md)
  - [alltoallv](./hccl_ops/alltoallv.md)
  - [alltoallvc](./hccl_ops/alltoallvc.md)
  - [send](./hccl_ops/send.md)
  - [receive](./hccl_ops/receive.md)
  - [Sample Code](./hccl_ops/sample_code.md)
    - [Code Example](./hccl_ops/code_example.md)
    - [Sample Running](./hccl_ops/sample_running.md)

- [Supported TensorFlow 1.15 APIs](support-list-tf-1-15.md)
