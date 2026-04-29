# NPUStrategy构造函数

## 功能说明

NPUStrategy类的构造函数。NPUStrategy继承了tf.distribute.Strategy类，可以调用基类的原生接口，用于在NPU环境中实现分布式训练。

## 函数原型

```python
class NPUStrategy(distribute_lib.StrategyV1):
    def __init__(self, device="/cpu:0")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| device | 输入 | 预留参数，当前仅支持配置为默认值"/cpu:0"，推荐不配置。 |

## 返回值

返回NPUStrategy类对象。

## 调用示例

构建NPUStrategy实例，在NPU环境实现分布式训练：

```python
from npu_bridge.npu_init import *
# ...
strategy = npu_strategy.NPUStrategy()
#使用strategy实现分布式训练。使用方式与tf.distribute.Strategy相同
# ...
```
