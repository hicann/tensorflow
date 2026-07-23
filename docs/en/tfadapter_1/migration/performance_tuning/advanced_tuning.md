# Advanced Tuning

To identify performance bottlenecks by analyzing profile data and take corresponding tuning measures, developers need to have rich tuning experience.

- If analysis of the iteration trace data shows that the  **FP to BP Time**  is long, analyze the time-consuming operators based on the  **op_statistic_\*.csv**  and  **op_summary_\*.csv**  files.

    If profile data analysis shows that the performance of the ReduceSum operator is poor, add the operator to the blocklist for mixed precision according to  [What Do I Do If Operator ReduceSum Has Poor Performance on a Network?](../faq/networktest_reducesum_low_performance.md).

    If AI CPU operators exist, further analyze whether their time segments can be covered by the execution time of AI Core operators. If not, replace the AI CPU operators with AI Core operators. For details, see  [Ascend C Operator Development](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).

- If analysis of the iteration trace data shows that  **Data Aug Bound**  or frontend computing is time-consuming, you can use the native Profiling tool of TensorFlow to identify computing-irrelevant operations and shield them to improve the single-iteration performance.
- If analysis of the  **op_statistic_\*.csv**  file shows that some operators take a long time, you can use the  [op_precision_mode](../../apiref/session_config/performance_tuning.md#op_precision_mode) parameter to specify the operators as high-performance ones to improve the operator performance.
