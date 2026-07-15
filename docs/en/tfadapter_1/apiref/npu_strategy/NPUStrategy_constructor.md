# NPUStrategy Constructor

## Description

Constructs an object of class  **NPUStrategy** **NPUStrategy**  inherits the  **tf.distribute.Strategy**  class and can call the native APIs of the base class to implement distributed training in the NPU environment.

## Prototype

```python
class NPUStrategy(distribute_lib.StrategyV1):
    def __init__(self, device="/cpu:0")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| device | Input | Reserved parameter. Currently, only the default value "/cpu:0" is supported. You are advised to leave it unconfigured. |

## Returns

An object of the  **NPUStrategy**  class

## Example

Build an instance of  **NPUStrategy**  to implement distributed training with NPU environments.

```python
from npu_bridge.npu_init import *
...
strategy = npu_strategy.NPUStrategy()
# Use strategy to implement distributed training. The usage is the same as that of tf.distribute.Strategy.
...
```
