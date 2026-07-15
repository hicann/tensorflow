# Profiler Constructor

## Description

Constructs an object of the  **Profiler**  class, which is used to enable the profiling function locally. For example, you can collect the profile data of a local subgraph on the TensorFlow network or a specified step.

## Prototype

```python
class Profiler(object):
    def __init__(
        self,
        *,
        level: str = "L0",
        aic_metrics: str = "",
        output_path: str = ""
    )
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| level | Input | Profiler level, which controls the range of data to be profiled.<br>  - L0: collects the task scheduling duration (task_time) and profile data related to the ACL API.<br>  - L1: collects the profile data of collective communication operators and AI Core operators in addition to that collected at L0.<br>  - L2: collects the profile data of runtime component execution and AI CPU operators in addition to that collected at L1.<br><br><br>Default value: L0 |
| aic_metrics | Input | When level is set to L1 or L2, you can use this parameter to collect the profile data of the AI Core and AI Vector Core hardware.<br><br>  - ArithmeticUtilization: time consumptions and percentages of Cube and Vector instructions.<br>  - PipeUtilization: percentages of time taken by compute units and MTEs.<br>  - Memory: memory read/write bandwidth rate.<br>  - MemoryL0: L0 read/write bandwidth rate.<br>  - MemoryUB: UB read/write bandwidth rate.<br>  - ResourceConflictRatio: ratio of pipeline queue instructions.<br>  - L2Cache: cache re-allocations upon missing of the read/write cache hit count.<br><br>For details about the collection items of each level, see "Profile Data File References > op_summary (Operator Details)" (Profiling Instructions).<br>Default value: PipeUtilization |
| output_path | Input | Path for storing profiling result files. The specified path must be created in advance in the environment (either in a container or on the host) where training is performed. The running user must have the read and write permissions on this path. The path can be an absolute path or a relative path (relative to the path where the command is executed). The path cannot contain the following special characters: "\n", "\f", "\r", "\b", "\t", "\v", and "\u007F".<br>  - An absolute path starts with a slash (/), for example, /home/test/output.<br>  - A relative path starts with a directory name, for example, output.<br>  - This parameter has a higher priority than the environment variable ASCEND_WORK_PATH. For details about ASCEND_WORK_PATH, see "Installation" in the [Environment Variables](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0001.html).<br><br><br>By default, this parameter is left blank. If it is left blank, the result files are stored in the current directory. |

## Returns

None

## Restrictions

- The  **Profiler**  class needs to be called using the  **with**  statement, and the profile data collection function takes effect in the corresponding scope.
- The  **Profiler**  class can be called only in session mode.
- The  **Profiler**  class cannot be nested.

    The following is an incorrect calling example:

    ```python
    with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
        sess.run(add)
    ```

- The  **Profiler**  class cannot be used together with parameters  **profiling_mode**  and  **profiling_options**  in  [Session Configuration](../session_config/Profiling.md), parameters  **enable_profiling**  and  **profiling_options**  in  [NPURunConfig Configuration](../npu_config/profilingconfig_constructor.md), and environment variables  **PROFILING_MODE**  and  **PROFILING_OPTIONS**. For details about the environment variables, see  _[Environment Variables](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/envvar/envref_07_0001.html)_.
- The  **Profiler**  class does not support multi-thread calling.

## Example

```python
import tensorflow as tf
from npu_bridge.npu_init import *

......
a = tf.placeholder(tf.int32, (None,None))
b = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
add = tf.add(a, b)

with tf.Session(config=session_config, graph=g) as sess:
  with profiler.Profiler(level="L1", aic_metrics=str("ArithmeticUtilization"), output_path = "./"):
    result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
```
