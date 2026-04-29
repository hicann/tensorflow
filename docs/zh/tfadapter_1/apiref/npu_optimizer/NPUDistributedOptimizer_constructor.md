# NPUDistributedOptimizer构造函数

## 功能说明

NPUDistributedOptimizer类的构造函数，用于包装用户提供的单机训练优化器，构造NPU分布式训练优化器。

支持单机单卡、单机多卡、多机多卡等组网形式下，各个Device之间计算梯度后执行梯度聚合操作。

## 函数原型

```python
class NPUDistributedOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer,
                 is_weight_update_sharding=False,
                 name=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| optimizer | 输入 | 用于梯度计算和更新权重的单机版训练优化器。 |
| is_weight_update_sharding | 输入 | bert网络分布式训练场景下，可以通过该参数，对weight/grad数据根据大小进行分组，每个device上仅更新对应分组的数据，缩短梯度更新阶段的计算时间，最终在save阶段再将各自的数据broadcast到其他所有节点。<br>通过这个方法可缩短梯度更新时间、减少内存占用。 |
| name | 输入 | 优化器名称。 |

## 返回值

返回NPUDistributedOptimizer类对象。

## 调用示例

用户定义单机版优化器后，再使用该优化器包装，举例如下：

```python
import tensorflow as tf
from npu_bridge.npu_init import *

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = NPUDistributedOptimizer(optimizer)
```
