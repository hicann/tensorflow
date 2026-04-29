# Profiler构造函数

## 功能说明

Profiler类的构造函数，用于局部打开Profiling功能，例如仅采集TensorFlow网络中局部子图的性能数据，或采集指定step的性能数据。

## 函数原型

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

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| level | 输入 | 开启Profiler的级别，用于控制采集性能数据的范围。<br>  - L0（默认值）：主要采集任务调度耗时信息（task_time）和acl接口执行相关的性能数据。<br>  - L1：在L0的采集数据基础上，额外采集集合通信算子性能数据和AI Core算子性能数据。<br>  - L2：在L1的采集数据基础上，额外采集Runtime组件执行相关的性能数据和AI CPU算子性能数据。 |
| aic_metrics | 输入 | 当level配置为“L1”或“L2”时，可通过此参数采集AI Core和AI Vector Core硬件相关的性能指标，包含如下取值：<br>  - ArithmeticUtilization：cube及vector类型指令耗时和占比。<br>  - PipeUtilization（默认值）：计算单元和搬运单元耗时占比。<br>  - Memory：内存读写带宽速率。<br>  - MemoryL0：L0读写带宽速率。<br>  - MemoryUB：UB读写带宽速率。<br>  - ResourceConflictRatio：流水线队列类指令占比。<br>  - L2Cache：读写cache命中次数和缺失后重新分配次数。<br>关于每一种取值包含的详细采集项及其含义可参见《[性能调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)》中的“性能数据文件参考 > op_summary（算子详细信息）”章节。 |
| output_path | 输入 | Profiling采集结果文件保存路径。该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且运行用户具有读写权限，支持配置为绝对路径或相对路径（相对执行命令行时的当前路径）。路径中不能包含特殊字符："\n"、"\f"、"\r"、"\b"、"\t"、"\v"、"\u007F"。<br>  - 绝对路径配置以“/”开头，例如：/home/test/output。<br>  - 相对路径配置直接以目录名开始，例如：output。<br>  - 该参数优先级高于环境变量ASCEND_WORK_PATH，关于ASCEND_WORK_PATH的详细说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“安装配置相关”章节。<br>默认值为空。此参数配置为空时，采集结果文件保存在当前目录下。 |

## 返回值

无

## 约束说明

- Profiler类需要通过with语句调用，性能数据采集功能会在对应的作用域内生效。
- Profiler类仅支持session模式调用。
- Profiler类不能嵌套使用。

    如下所示，是错误的调用方法。

    ```python
    with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
        sess.run(add)
    ```

- Profiler类不能与[session配置](../session配置.md#profiling)中的参数“profiling_mode”、“profiling_options”，[NPURunConfig配置](../npu_config/profilingconfig_constructor.md)中的参数“enable_profiling”、“profiling_options”，以及环境变量“PROFILING_MODE”、“PROFILING_OPTIONS”同时使用，关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
- Profiler类不支持多线程调用。

## 调用示例

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
