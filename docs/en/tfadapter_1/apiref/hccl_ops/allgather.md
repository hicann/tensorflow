# allgather

## Description

Re-sorts the inputs of all ranks in the communicator by rank ID, combines the inputs, and sends the results to the outputs of all ranks.

![](../figures/allgather.png)

> [!NOTE]NOTE
> For the AllGather operation, each rank receives a set of data that is resorted by rank ID, that is, AllGather outputs of all ranks are the same.

## Prototype

```python
def allgather(tensor, rank_size, group="hccl_world_group", fusion=0, fusion_id=-1)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tensor | Input | TensorFlow tensor type.<br>For the Ascend 910_95 AI Processor, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, and bfp16.<br>For the Atlas A3 training products/Atlas A3 inference products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas A2 training products/Atlas A2 inference products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas training products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64.<br>For the Atlas 300I Duo inference card, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64. |
| rank_size | Input | Int type.<br>Number of devices in a group.<br>The maximum value is 32768. |
| group | Input | A string containing a maximum of 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |
| fusion | Input | Int type.<br>AllGather operator fusion flag. The value can be one of the following:<br><br>  - 0: The AllGather operator is not fused with other AllGather operators during network compilation.<br>  - 2: AllGather operators with the same fusion_id are fused during network compilation. |
| fusion_id | Input | Int type.<br>AllGather operator fusion ID.<br>When fusion is set to 2, AllGather operators with the same fusion_id are fused during network compilation. |

## Returns

The result tensor

## Restrictions

The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.

## Example

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
rank_size = 2
result = hccl_ops.allgather(tensor, rank_size)
```
