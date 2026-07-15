# allreduce

## Description

Performs the reduction operation on the input data of all ranks in a group and sends the result to the output buffer of all ranks. The reduction operation type is specified by the  **reduction**  parameter. This API operates the collective communication operator AllReduce.

![](../figures/allreduce.png)

## Prototype

```python
def allreduce(tensor, reduction, fusion=1, fusion_id=-1, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tensor | Input | TensorFlow tensor type.<br>Ascend 950PR/Ascend 950DT: The supported data types are int8, int16, int32, int64, uint64, float16, float32, float64, and bfp16. Data types int64, uint64, and float64 supports only intra-node communication.<br>Atlas A3 training product/Atlas A3 inference product: The supported data types are int8, int16, int32, int64, float16, float32, and bfp16.<br>Atlas A2 training product/Atlas A2 inference product: The supported data types are int8, int16, int32, int64, float16, float32, and bfp16. Note that the performance will deteriorate for the int64 data type.<br>Atlas training product: The supported data types are int8, int32, int64, float16, and float32.<br>Atlas 300I Duo Inference Card: The supported data types are int8, int16, int32, float16, and float32. |
| reduction | Input | Reduction operation, string type.<br>Ascend 950PR/Ascend 950DT: The supported operation types are sum, max, and min.<br>Atlas A3 training product/Atlas A3 inference product: The supported operation types are sum, max, min, and prod. In the current version, the prod operation does not support the int16 or bfp16 data type.<br>Atlas A2 training product/Atlas A2 inference product: The supported operation types are sum, max, min, and prod. In the current version, the prod operation does not support the int16 or bfp16 data type.<br>Atlas 300I Duo Inference Card: The supported operation types are sum, max, min, and prod. In the current version, the max, min, and prod operations do not support the int16 data type. |
| fusion | Input | AllReduce operator fusion flag, int type. The value can be one of the following:<br><br>  - 0: The AllReduce operator is not fused with other AllReduce operators during network compilation.<br>  - 1: The AllReduce operator is fused based on the gradient splitting policy during network compilation.<br>  - 2: AllReduce operators with the same fusion_id are fused during network compilation. |
| fusion_id | Input | AllReduce operator fusion ID, int type.<br>When fusion is set to 2, AllReduce operators with the same fusion_id are fused during network compilation. |
| group | Input | A string containing a maximum of 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

The result tensor after the  **allreduce**  operation is performed on the input tensor.

## Restrictions

- The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.
- Each rank can have only one input.
- The upstream node of  **allreduce**  must not be  **variable**.
- The input tensor size must be less than or equal to 8 GB.
- For the AllReduce operator fusion, only the reduction type  **sum**  is supported.

## Example

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
result = hccl_ops.allreduce(tensor, "sum")
```
