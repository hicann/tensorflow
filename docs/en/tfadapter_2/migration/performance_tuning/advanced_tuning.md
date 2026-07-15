# Advanced Tuning

Identifying performance bottlenecks through profile data analysis and applying corresponding tuning measures requires extensive tuning experience.

- If analysis of the iteration trace data shows that the  **FP to BP Time**  is long, analyze the time-consuming operators based on the  **op_statistic_\*.csv**  and  **op_summary_\*.csv**  files.

    If profile data analysis shows that the performance of the ReduceSum operator is poor, add the operator to the blocklist for mixed precision according to  [What Do I Do If Operator ReduceSum Has Poor Performance on a Network?](../faq/networktest_reducesum_low_performance.md).

    If AI CPU operators exist, check whether their execution time segments can be covered by the execution time of AI Core operators. If no, implement the AI CPU operators in AI Core operator mode. For details, see  [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommercialOpdevAscendC).

- If analysis of the iteration trace data shows that  **Data Aug Bound**  or frontend computing is time-consuming, you can use the native Profiling tool of TensorFlow to identify computing-irrelevant operations and shield them to improve the single-iteration performance.
- If analysis of the  **op_statistic_\*.csv**  file shows that some operators take a long time, you can use the  [op_precision_mode](../../apiref/npu-global_options/performance_tuning.md#op_precision_mode)  parameter to specify the operators as high-performance ones to improve the operator performance.
