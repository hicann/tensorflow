# npu_distributed_optimizer_wrapper

## Description

Adds the AllReduce operation of NPU to the input gradient function of the optimizer, combines them into one function, and returns the optimizer. This API is used only in distributed scenarios.

## Prototype

```python
def npu_distributed_optimizer_wrapper(optimizer)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| optimizer | Input | TensorFlow gradient training optimizer. |

## Returns

TensorFlow gradient training optimizer.

## Example

```python
from npu_bridge.npu_init import *
# TF scenario
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Use the SGD optimizer.
optimizer = npu_distributed_optimizer_wrapper(optimizer) # Use NPU-based distributed computing to update gradients.
# Keras scenario
optimizer = tf.keras.optimizers.SGD()
optimizer = npu_distributed_optimizer_wrapper(optimizer) # Use NPU-based distributed computing to update gradients.
```
