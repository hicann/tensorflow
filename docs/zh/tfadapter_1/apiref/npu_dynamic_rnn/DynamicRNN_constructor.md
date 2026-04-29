# DynamicRNN构造函数

## 功能说明

TensorFlow侧使用该接口，支持RNN类网络训练、推理。

## 函数原型

```python
class DynamicRNN(_DynamicBasic):
    def __init__(self,
                 hidden_size,
                 dtype,
                 cell_type="LSTM",
                 direction=DYNAMIC_RNN_UNIDIRECTION,
                 cell_depth=1,
                 use_peephole=False,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation="tanh",
                 forget_bias=0.0,
                 is_training=True)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| hidden_size | 输入 | RNN模型中隐状态h的维度。 |
| dtype | 输入 | weight、bias初始化的数据类型，注：传入的数据最终会转化成D支持的类型。 |
| cell_type | 输入 | （可选）目前仅支持"LSTM"。 |
| direction | 输入 | （可选）目前仅支持DYNAMIC_RNN_UNIDIRECTION。 |
| cell_depth | 输入 | （可选）目前仅支持单层。 |
| use_peephole | 输入 | （可选）目前不支持窥孔计算。 |
| keep_prob | 输入 | （可选）目前不支持dropout。 |
| cell_clip | 输入 | （可选）目前不支持数值裁剪。 |
| num_proj | 输入 | （可选）目前不支持投影计算。 |
| time_major | 输入 | （可选）目前仅支持输入x是【num_step, batch_size, embedding】模式。 |
| activation | 输入 | （可选）目前仅支持"tanh"。 |
| forget_bias | 输入 | （可选）默认是0.0。 |
| is_training | 输入 | （可选）默认是训练模式。 |

## 返回值

- output_y：RNN的输出tensor，shape为【num_step, batch_size, hidden_size】。
- output_h：RNN的输出tensor，shape为【num_step, batch_size, hidden_size】。
- output_c：RNN的输出tensor，shape为【num_step, batch_size, hidden_size】。
- i：RNN计算的中间结果，用于反向计算使用。
- j：RNN计算的中间结果，用于反向计算使用。
- f：RNN计算的中间结果，用于反向计算使用。
- o：RNN计算的中间结果，用于反向计算使用。
- tanhc：RNN计算的中间结果，用于反向计算使用。

## 使用约束

该接口为tf.nn.dynamic_rnn的高性能实现接口，用户可以对原生接口进行替换，需要注意的是：

- TensorFlow原生接口tf.nn.dynamic_rnn可以对cell层做添加处理，例如首尾连接等，DynamicRNN不支持直接对cell层做添加处理，需要在函数外自行实现。
- 目前该接口支持的功能有限，具体约束详见参数说明表。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicRNN
inputs = tf.random.normal(shape=(25, 64, 128))
lstm = DynamicRNN(hidden_size=16, dtype=tf.float16, is_training=False)
y, output_h, output_c, i, j, f, o, tanhc = lstm(inputs, seq_length=None, init_h=None, init_c=None)
```
