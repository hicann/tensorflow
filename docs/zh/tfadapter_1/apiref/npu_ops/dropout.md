# dropout

## 功能说明

和tf.nn.dropout功能相同。以概率keep_prob（保留概率）将输入Tensor中的元素置零，未被丢弃的元素值按“1/keep_prob”缩放，最终输出Tensor的shape与输入Tensor保持一致。

## 函数原型

```python
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 输入Tensor，float类型。 |
| keep_prob | 输入 | 标量Tensor，float类型。代表每个元素保留的概率。 |
| noise_shape | 输入 | 一维Tensor，int32类型。表示随机生成的keep_drop标志的形状。 |
| seed | 输入 | 随机数种子。 |
| name | 输入 | 网络层的名称。 |

## 返回值

tensor：对输入x执行完dropout操作之后的输出Tensor。

## 调用示例

```python
from npu_bridge.npu_init import *
layers = npu_ops.dropout()
```
