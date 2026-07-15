# npu_dynamic_rnn

## Description

Creates a high-performance neural network specified by RNNCell.

## Prototype

```python
def npu_dynamic_rnn(cell,
                    inputs,
                    initial_state=None,
                    dtype=None,
                    sequence_length=None,
                    scope=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| cell | Input | RNNCell instance, which is the memory unit of long short-term memory (LSTM) and gated recurrent unit (GRU). |
| inputs | Input | An input list whose length is T. Each input is a tuple whose shape is [max_time, batch_size, input_size], or a nested tuple of this shape. |
| initial_state | Input | (Optional) Initial state of the recurrent neural network (RNN). If cell.state_size is an integer, it must be a tensor whose shape is [batch_size, cell.state_size]. If cell.state_size is a tuple, it must be a tuple of the [batch_size, s] tensor in cell.state_size. |
| dtype | Input | (Optional) Data type of the initial state and expected output. This parameter is required if initial_state is empty, or there is a heterogeneous data type in the RNN state. |
| sequence_length | Input | Length of each sequence for an input. This parameter is an int32 or int64 vector (tensor) whose size is [batch_size]. The value range is [0, T). |
| scope | Input | VariableScope of the subgraph.<br>Defaults to "rnn". |

## Returns

Result tensor of the RNN.

Final state.

## Restrictions

This API applies to the neural machine translation \(NMT\) network training script in the  **while_loop**  expansion scenario.

## Example

```python
from npu_bridge.npu_init import *
# Original code:
inputs = npu_unstack(self.encoder_emb_inp, axis=0)
encoder_outputs , encoder_state = static_rnn(
    cell,
    inputs,
    dtype= dtype,
    sequence_length = sequence_length
     )
encoder_outputs = npu_stack( encoder_outputs, axis=0 )
# Replace it with:
encoder_outputs , encoder_state = npu_rnn.npu_dynamic_rnn(
    cell,
    inputs=self.encoder_emb_inp,
    dtype= dtype,
    sequence_length= sequence_length)
```
