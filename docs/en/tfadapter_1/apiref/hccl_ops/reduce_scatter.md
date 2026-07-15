# reduce_scatter

## Function

Functions as the operation API of the ReduceScatter operator to evenly divide the input data of all ranks in a communicator into  **rank size**  parts, perform reduction \(sum, prod, max, and min\) on 1/**rank size**  part of data of each rank, and distributes the result to the output buffer of each rank based on the number.

![](../figures/reduce_scatter.png)

## Prototype

```python
def reduce_scatter(tensor, reduction, rank_size, group="hccl_world_group", fusion=0, fusion_id=-1)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tensor | Input | TensorFlow tensor type.<br>Ascend 950PR/Ascend 950DT: The supported data types are int8, int16, int32, int64, uint64, float16, float32, float64, and bfp16. Data types int64, uint64, and float64 supports only intra-node communication.<br>Atlas A3 training product/Atlas A3 inference product: The supported data types are int8, int16, int32, int64, float16, float32, and bfp16.<br>Atlas A2 training product/Atlas A2 inference product: The supported data types are int8, int16, int32, int64, float16, float32, and bfp16. Note that the performance will deteriorate for the int64 data type.<br>Atlas training product: The supported data types are int8, int32, int64, float16, and float32.<br>Atlas 300I Duo Inference Card: The supported data types are int8, int16, int32, float16, and float32.<br>Note that the size of the first dimension of a tensor must be an integer multiple of the rank size. |
| reduction | Input | Reduction operation, string type.<br>Ascend 950PR/Ascend 950DT: The supported operation types are sum, max, and min.<br>Atlas A3 training product/Atlas A3 inference product: The supported operation types are sum, max, min, and prod. In the current version, the prod operation does not support the int16 or bfp16 data type.<br>Atlas A2 training product/Atlas A2 inference product: The supported operation types are sum, max, min, and prod. In the current version, the prod operation does not support the int16 or bfp16 data type.<br>Atlas 300I Duo Inference Card: The supported operation types are sum, max, min, and prod. In the current version, the max, min, and prod operations do not support the int16 data type. |
| rank_size | Input | Number of devices in a group, int type.<br>Maximum value: 32768. |
| group | Input | A string containing a maximum of 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |
| fusion | Input | reducescatter operator fusion flag, int type. The value can be one of the following:<br><br>  - 0: The ReduceScatter operator is not fused with other ReduceScatter operators during network compilation.<br>  - 2: ReduceScatter operators with the same fusion_id are fused during network compilation. |
| fusion_id | Input | reducescatter operator fusion ID, int type.<br>If fusion is set to 2, ReduceScatter operators with the same fusion_id are fused during network compilation. |

## Returns

The result tensor after the  **reducescatter**  operation is performed on the input tensor.

## Restrictions

- The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.
- The input tensor size must be less than or equal to 8 GB.
- For the reducescatter operator fusion, only the reduction type  **sum**  is supported.

## Example

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((2, 3), minval=1, maxval=10, dtype=tf.float32)
rank_size = 2
result = hccl_ops.reduce_scatter(tensor, "sum", rank_size)
```
