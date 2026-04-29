# LARSV2

## 功能说明

该算子基于权重的范数和梯度的范数在不同层级上使用不同的学习率，对梯度缩放。通常用于提升大batch size场景下的训练精度，用于大规模集群训练，减少训练时间。

## 函数原型

```python
def LARSV2(input_weight,
           input_grad,
           weight_decay,
           learning_rate,
           hyperpara=0.001,
           epsilon=0.00001,
           use_clip=False,
           name=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| input_weight | 输入 | 权重Tensor，为float类型。 |
| input_grad | 输入 | 权重梯度Tensor，为float类型。 |
| weight_decay | 输入 | 标量Tensor，为float类型。 |
| learning_rate | 输入 | 标量Tensor，为float类型，表示学习率。 |
| hyperpara | 输入 | 标量，算子的超参，为float类型，一般设定为0.001。 |
| epsilon | 输入 | 标量，一般为很小的正数，防止分母为0，一般设定为1e-5。 |
| use_clip | 输入 | bool类型，默认为False。<br>当配置为True时，表示缩放系数需要限定在一定的范围内。 |
| name | 输入 | 网络层的名称。 |

## 返回值

tensor：对输入的梯度进行更新后的输出梯度Tensor。

## 调用示例

```python
from npu_bridge.npu_init import *
layers = npu_ops.LARSV2(input_weight , input_grad, weight_decay, learning_rate)
```
