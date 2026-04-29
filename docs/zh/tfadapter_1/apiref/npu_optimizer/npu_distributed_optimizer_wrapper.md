# npu_distributed_optimizer_wrapper

## 功能说明

对传入的optimizer中的求梯度的函数添加NPU的allreduce操作之后，将包含原生优化器求梯度和NPU的allreduce两个操作合并为一个函数，替换原生优化器的求梯度的函数，最终返回输入的优化器。该接口仅在分布式场景下使用。

## 函数原型

```python
def npu_distributed_optimizer_wrapper(optimizer)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| optimizer | 输入 | TensorFlow梯度训练优化器。 |

## 返回值

返回输入的optimizer。

## 调用示例

```python
from npu_bridge.npu_init import *
# tf场景
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # 使用SGD优化器
optimizer = npu_distributed_optimizer_wrapper(optimizer) # 使用NPU分布式计算，更新梯度
# keras场景
optimizer = tf.keras.optimizers.SGD()
optimizer = npu_distributed_optimizer_wrapper(optimizer) # 使用NPU分布式计算，更新梯度
```
