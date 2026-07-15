# npu_allreduce

## Description

Performs AllReduce and update operations on gradients after the gradient computing is complete.

## Prototype

```python
def _npu_allreduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| values | Input | Tensor list or tensor. |
| reduction | Input | Op type of Reduce. The value can be sum or mean. |
| fusion | Input | Operator fusion flag, which is of the int type.<br><br>  - 0: disabled. The AllReduce operator is not fused with other AllReduce operators.<br>  - 1 (default): enabled. The AllReduce operator is fused based on the gradient splitting strategy.<br>  - 2: enabled. AllReduce operators with the same fusion_id are fused. |
| fusion_id | Input | Operator fusion index flag. AllReduce operators with the same fusion_id will be fused. |
| group | Input | Group name, which is of the string type. It can be a user-defined value or hccl_world_group. |

## Returns

Tensor list or tensor, which is consistent with the inputs.

## Example

```python
from npu_bridge.npu_init import *
grads = npu_allreduce(tf.gradients(a + b, [a, b], stop_gradients=[a, b]))
```
