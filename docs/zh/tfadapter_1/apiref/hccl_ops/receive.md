# receive

## 功能说明

提供group内点对点通信数据的receive功能。

## 函数原型

```python
def receive(shape, data_type, sr_tag, src_rank, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | 接收tensor的shape。 |
| data_type | 输入 | 接收数据的数据类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、 int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。<br>针对Atlas 300I Duo 推理卡，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。 |
| sr_tag | 输入 | 消息标签，相同sr_tag的send/recv对可以收发数据，int类型。 |
| src_rank | 输入 | 接收数据的源节点，该rank是group中的rank id，int类型。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

进行receive操作之后从对端接收到的tensor。

## 约束说明

- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- send和receive必须配对使用，即调用send接口后，需要等到与之配对的receive接口接收数据后，才可以进行下一个接口调用。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
sr_tag = 0
src_rank = 0
tensor = hccl_ops.receive(tensor.shape, tensor.dtype, sr_tag, src_rank)
```
