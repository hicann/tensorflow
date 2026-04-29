# npu_dynamic_rnn

## 功能说明

创建由RNNCell指定的高性能神经网络。

## 函数原型

```python
def npu_dynamic_rnn(cell,
                    inputs,
                    initial_state=None,
                    dtype=None,
                    sequence_length=None,
                    scope=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cell | 输入 | RNNCell的实例，为LSTM、GRU等的记忆单元。 |
| inputs | 输入 | 长度为T的输入列表，每一个都是shape为[max_time，batch_size，input_size]或此类元素的嵌套元组。|
| initial_state | 输入 | （可选）RNN的初始状态。如果cell.state_size是整数，则必须是shape为[batch_size，cell.state_size]的Tensor；如果cell.state_size是一个元组，则它应该是cell.state_size中形状为[batch_size，s]的张量的元组。 |
| dtype | 输入 | （可选）初始状态和预期输出的数据类型。如果initial_state为空或RNN状态具有异构dtype，则该参数为必需。 |
| sequence_length | 输入 | 指定输入中每个序列的长度。一个int32或int64向量（张量）大小为[batch_size]，值为[0，T）。|
| scope | 输入 | 创建子图的VariableScope。<br>默认为“rnn”。 |

## 返回值

tensor：RNN的输出Tensor。

state：最终状态。

## 约束说明

适用于NMT网络训练脚本的while_loop循环展开场景。

## 调用示例

```python
from npu_bridge.npu_init import *
# 原代码：
inputs = npu_unstack(self.encoder_emb_inp, axis=0)
encoder_outputs , encoder_state = static_rnn(
    cell,
    inputs,
    dtype= dtype,
    sequence_length = sequence_length
     )
encoder_outputs = npu_stack( encoder_outputs, axis=0 )
# 替换成：
encoder_outputs , encoder_state = npu_rnn.npu_dynamic_rnn(
    cell,
    inputs=self.encoder_emb_inp,
    dtype= dtype,
    sequence_length= sequence_length)
```
