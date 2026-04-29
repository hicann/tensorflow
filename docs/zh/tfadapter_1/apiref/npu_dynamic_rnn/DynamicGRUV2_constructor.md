# DynamicGRUV2构造函数

## 功能说明

TensorFlow侧使用该接口，支持RNN类网络训练、推理。

## 函数原型

```python
class DynamicGRUV2(_DynamicBasic):
    def __init__(self,
                 hidden_size,
                 dtype,
                 direction=DYNAMIC_RNN_UNIDIRECTION,
                 cell_depth=1,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation="tanh",
                 gate_order="zrh",
                 reset_after=True,
                 is_training=True)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| hidden_size | 输入 | GRU模型中隐状态h的维度。 |
| dtype | 输入 | weight、bias初始化的数据类型，注：传入的数据最终会转化成D支持的类型。 |
| direction | 输入 | （可选）目前仅支持DYNAMIC_RNN_UNIDIRECTION。 |
| cell_depth | 输入 | （可选）目前仅支持单层。 |
| keep_prob | 输入 | （可选）目前不支持dropout。 |
| cell_clip | 输入 | （可选）目前不支持数值裁剪。 |
| num_proj | 输入 | （可选）目前不支持投影计算。 |
| time_major | 输入 | （可选）目前仅支持输入x是【num_step, batch_size, embedding】模式。 |
| activation | 输入 | （可选）目前仅支持“tanh”。 |
| gate_order | 输入 | （可选）表示几个门的顺序，默认为“zrh”，常用的另一个顺序为“rzh”。 |
| reset_after | 输入 | （可选）默认为TRUE，表示矩阵乘法之后将重置门应用到隐藏状态 。 |
| is_training | 输入 | （可选）默认是训练模式。 |

## 返回值

output_y：RNN的输出tensor，shape为【num_step, batch_size, hidden_size】。

output_h：RNN的输出tensor，shape为【num_step, batch_size, hidden_size】。

update：RNN计算的中间结果，用于反向计算使用。

reset：RNN计算的中间结果，用于反向计算使用。

new：RNN计算的中间结果，用于反向计算使用。

hidden_new：RNN计算的中间结果，用于反向计算使用。

## 使用约束

目前该接口支持的功能有限，具体约束详见参数说明表。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicGRUV2
inputs = tf.random.normal(shape=(25, 64, 128))
gru = DynamicGRUV2(hidden_size=16, dtype=tf.float16, is_training=False)
y, output_h, update, reset, new, hidden_new = gru(inputs, seq_length=None, init_h=None)
```
