# dropout

## Description

It has the same functionality as  **tf.nn.dropout**. Elements of the input tensor are randomly set to zero with a probability of  **1 – keep_prob**. The remaining elements are scaled by a factor of  **1/keep_prob**  to ensure that the output tensor maintains the same shape as the input tensor.

## Prototype

```python
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| x | Input | Input tensor of type float. |
| keep_prob | Input | Scalar tensor of type float, which indicates the retention probability of each element. |
| noise_shape | Input | 1D tensor of type int32, which indicates the shape of the randomly generated keep_drop flag. |
| seed | Input | Random seed. |
| name | Input | Name of the network layer. |

## Returns

Result tensor after the dropout operation is performed on input  **x**.

## Example

```python
from npu_bridge.npu_init import *
layers = npu_ops.dropout()
```
