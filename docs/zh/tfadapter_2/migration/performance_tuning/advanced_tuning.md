# 进阶调优

通过Profiling数据分析出性能瓶颈点，再进行对应的调优手段，需要开发人员有丰富的调优经验，对开发人员要求较高。

- 通过对迭代轨迹数据进行分析，若发现“FP to BP Time”耗时较长，可进一步根据op_statistic_\*.csv与op_summary_\*.csv文件，分析耗时较长的算子。

    若分析算子性能数据时，发现ReduceSum算子的性能很差，可参见[网络调测时ReduceSum算子性能差](../faq/networktest_reducesum_low_performance.md)将算子配置到混合精度黑名单进行处理。

    若存在AI CPU算子，需进一步分析其执行时间段是否能够被AI Core算子的执行时间所覆盖。如若不能，则应考虑将该AI CPU算子通过AI Core算子的方式重新实现，详细的实现方法可参见《Ascend C算子开发指南》。

- 通过对迭代轨迹数据进行分析，若发现“Data Aug Bound”即前端计算耗时较长，可进一步结合TensorFlow原生的Profiling工具，识别出计算无关操作，并将其进行屏蔽，从而提升单轮迭代性能。
- 分析op_statistic_\*.csv文件，发现某些算子耗时较长，可尝试使用“[op_precision_mode](../../apiref/npu-global_options/performance_tuning.md#op_precision_mode)”参数指定耗时较长的算子为高性能模式，从而提升算子性能。
