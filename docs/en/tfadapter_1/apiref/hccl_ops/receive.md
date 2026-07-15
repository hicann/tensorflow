# receive

## Description

Receives data from a rank within a collective communication group.

## Prototype

```python
def receive(shape, data_type, sr_tag, src_rank, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| shape | Input | Shape of the received tensor. |
| data_type | Input | Data type of the received data.<br>For the Ascend 950PR/Ascend 950DT, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>Atlas A3 training product/Atlas A3 inference product: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>Atlas A2 training product/Atlas A2 inference product: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas training product, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64.<br>Atlas 300I Duo Inference Card: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64. |
| sr_tag | Input | Message tag. The send/recv pairs with the same sr_tag can receive and send data, int type. |
| src_rank | Input | Source rank of the received data. This rank indicates the rank ID in the group, int type. |
| group | Input | A string containing a maximum of 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

The result tensor.

## Restrictions

- The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.
- The send and receive APIs must be used in pairs. That is, after the send API is called, the next API can be called only after the paired receive API receives data.

## Example

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
sr_tag = 0
src_rank = 0
tensor = hccl_ops.receive(tensor.shape, tensor.dtype, sr_tag, src_rank)
```
