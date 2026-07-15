# send

## Function Description

Sends data to a rank within a collective communication group.

## Function Prototype

```python
def send(tensor, sr_tag, dest_rank, group="hccl_world_group")
```

## Parameters

| Option | Input/Output | Description |
| --- | --- | --- |
| tensor | Input | TensorFlow tensor type.<br>For the Ascend 910_95 AI Processor, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, and bfp16.<br>For the Atlas A3 training products/Atlas A3 inference products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas A2 training products/Atlas A2 inference products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas training products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64.<br>For the Atlas 300I Duo inference card, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64. |
| sr_tag | Input | Int type.<br>Message tag. The send/recv pairs with the same sr_tag can receive and send data. |
| dest_rank | Input | Int type.<br>Destination rank of the data. This rank indicates the rank ID in the group. |
| group | Input | A string containing a maximum of 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

The result tensor

## Constraints

- The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.
- The send and receive APIs must be used in pairs. That is, after the send API is called, the next API can be called only after the paired receive API receives data.

## Example

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
sr_tag = 0
dest_rank = 1
result = hccl_ops.send(tensor, sr_tag, dest_rank)
```
