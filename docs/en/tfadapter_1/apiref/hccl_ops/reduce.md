# reduce

## Function

Performs the sum operation \(or other reduction operations\) on the data of all ranks and sends the result to the specified position on the root rank.

![](../figures/reduce.png)

## Prototype

```python
def reduce(tensor, reduction, root_rank, fusion=0, fusion_id=-1, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tensor | Input | TensorFlow tensor type.<br>Ascend 950PR/Ascend 950DT: The supported data types are int8, int16, int32, int64, uint64, float16, float32, float64, and bfp16. Data types int64, uint64, and float64 supports only intra-node communication.<br>Atlas A3 training product/Atlas A3 inference product: The supported data types are int8, int16, int32, int64, float16, float32, and bfp16.<br>Atlas A2 training product/Atlas A2 inference product: The supported data types are int8, int16, int32, int64, float16, float32, and bfp16. Note that the performance will deteriorate for the int64 data type.<br>Atlas training product: The supported data types are int8, int32, int64, float16, and float32. |
| reduction | Input | Reduction operation, string type.<br>Ascend 950PR/Ascend 950DT: The supported operation types are sum, max, and min.<br>Atlas A3 training product/Atlas A3 inference product: The supported operation types are sum, max, min, and prod. In the current version, the prod operation does not support the int16 or bfp16 data type.<br>Atlas A2 training product/Atlas A2 inference product: The supported operation types are sum, max, min, and prod. In the current version, the prod operation does not support the int16 or bfp16 data type. |
| root_rank | Input | Rank ID of the root rank, and must be a rank ID in the group, int type. |
| fusion | Input | reduce operator fusion flag, int type. The value can be one of the following:<br><br>  - 0: disabled. The Reduce operator is not fused with other Reduce operators.<br>  - 2: enabled. Operators with the same fusion_id are fused. |
| fusion_id | Input | reduce operator fusion ID, int type.<br>If fusion is set to 2, Reduce operators with the same fusion_id are fused during network compilation. |
| group | Input | A string containing a maximum of 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

The result tensor after the  **reduce**  operation is performed on the input tensor.

## Restrictions

- The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.
- The input tensor size must be less than or equal to 8 GB.
- For the reduce operator fusion, only the reduction type  **sum**  is supported.

## Example

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
result = hccl_ops.reduce(tensor, "sum", 0)
```
