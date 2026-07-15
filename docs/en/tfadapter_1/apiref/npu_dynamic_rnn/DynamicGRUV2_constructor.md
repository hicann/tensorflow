# DynamicGRUV2 Constructor

## Description

Used for RNN training and inference with TensorFlow.

## Prototype

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

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| hidden_size | Input | Dimension of the GRU hidden state h. |
| dtype | Input | Data type of initialized weight and bias. Note: The input data will be converted into the type supported by DynamicRNN. |
| direction | Input | (Optional) Only DYNAMIC_RNN_UNIDIRECTION is supported. |
| cell_depth | Input | (Optional) Only one layer is supported. |
| keep_prob | Input | (Optional) Dropout is not supported. |
| cell_clip | Input | (Optional) Cell state clipping is not supported. |
| num_proj | Input | (Optional) Projection calculation is not supported. |
| time_major | Input | (Optional) Only input x in [num_step, batch_size, embedding] is supported. |
| activation | Input | (Optional) Only "tanh" is supported. |
| gate_order | Input | (Optional) Gate order. Defaults to "zrh". "rzh" is also used commonly. |
| reset_after | Input | (Optional) Defaults to TRUE, indicating that the reset gate is applied to the hidden state after matrix multiplication. |
| is_training | Input | (Optional) By default, the training mode is used. |

## Returns

**output_y**: RNN output tensor, with the shape of \[num_step, batch_size, hidden_size\].

**output_h**: RNN output tensor, with the shape of \[num_step, batch_size, hidden_size\].

**update**: RNN intermediate result, which is used for backward propagation.

**reset**: RNN intermediate result, which is used for backward propagation.

**new**: RNN intermediate result, which is used for backward propagation.

**hidden_new**: RNN intermediate result, which is used for backward propagation.

## Restrictions

Currently, there are restrictions on the API. See the following table for details.

## Example

```python
import tensorflow as tf
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicGRUV2

inputs = tf.random.normal(shape=(25, 64, 128))
gru = DynamicGRUV2(hidden_size=16, dtype=tf.float16, is_training=False)
y, output_h, update, reset, new, hidden_new = gru(inputs, seq_length=None, init_h=None)
```
