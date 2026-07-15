# NPUDistributedOptimizer Constructor

## Description

Constructs an object of class  **NPUDistributedOptimizer**, which wraps around a single-server training optimizer to an NPU distributed training optimizer.

In single-server single-device, single-server multi-device, and multi-server multi-device networking modes, gradient aggregation can be performed among devices after gradient calculation.

## Prototype

```python
class NPUDistributedOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer,
                 is_weight_update_sharding=False,
                 name=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| optimizer | Input | Single-server training optimizer for gradient calculation and weight update. |
| is_weight_update_sharding | Input | Weight/Gradient update sharding by size. For distributed training with the BERT network, this parameter can be used to group weight and gradient data by size. Only the data of the corresponding group will be updated on each device, which shortens the compute time at the gradient update phase. Finally, the data is broadcast to all other nodes at the save phase.<br>This method can speed up gradient update and save memory footprint. |
| name | Input | Name of the optimizer. |

## Returns

An object of the  **NPUDistributedOptimizer**  class

## Example

After defining a single-server training optimizer, you can use it for wrapping. The following is an example:

```python
import tensorflow as tf
from npu_bridge.npu_init import *

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = NPUDistributedOptimizer(optimizer)
```
