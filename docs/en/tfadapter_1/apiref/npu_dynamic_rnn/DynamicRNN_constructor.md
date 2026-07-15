# DynamicRNN Constructor

## Description

Used for RNN training and inference with TensorFlow.

## Prototype

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

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| hidden_size | Input | Dimension of the RNN hidden state h. |
| dtype | Input | Data type of initialized weight and bias. Note: The input data will be converted into the type supported by DynamicRNN. |
| cell_type | Input | (Optional) Only "LSTM" is supported. |
| direction | Input | (Optional) Only DYNAMIC_RNN_UNIDIRECTION is supported. |
| cell_depth | Input | (Optional) Only one layer is supported. |
| use_peephole | Input | (Optional) Peephole connections are not supported. |
| keep_prob | Input | (Optional) Dropout is not supported. |
| cell_clip | Input | (Optional) Cell state clipping is not supported. |
| num_proj | Input | (Optional) Projection calculation is not supported. |
| time_major | Input | (Optional) Only input x in [num_step, batch_size, embedding] is supported. |
| activation | Input | (Optional) Only "tanh" is supported. |
| forget_bias | Input | (Optional) Defaults to 0.0. |
| is_training | Input | (Optional) By default, the training mode is used. |

## Returns

- **output_y**: RNN output tensor, with the shape of \[num_step, batch_size, hidden_size\].
- **output_h**: RNN output tensor, with the shape of \[num_step, batch_size, hidden_size\].
- **output_c**: RNN output tensor, with the shape of \[num_step, batch_size, hidden_size\].
- **i**: RNN intermediate result, which is used for backward propagation.
- **j**: RNN intermediate result, which is used for backward propagation.
- **f**: RNN intermediate result, which is used for backward propagation.
- **o**: RNN intermediate result, which is used for backward propagation.
- **tanhc**: RNN intermediate result, which is used for backward propagation.

## Restrictions

This API is the high-performance equivalent of  **tf.nn.dynamic_rnn**. You are advised to use this API instead. Note the following:

- The native TensorFlow API  **tf.nn.dynamic_rnn**  can add connections to cells \(such as connecting the output layer to the input layer\).  **DynamicRNN**  has no such native support and connection addition needs to be implemented separately.
- Currently, there are restrictions on the API. See the following table for details.

## Example

```python
import tensorflow as tf
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicRNN

inputs = tf.random.normal(shape=(25, 64, 128))
lstm = DynamicRNN(hidden_size=16, dtype=tf.float16, is_training=False)
y, output_h, output_c, i, j, f, o, tanhc = lstm(inputs, seq_length=None, init_h=None, init_c=None)
```
