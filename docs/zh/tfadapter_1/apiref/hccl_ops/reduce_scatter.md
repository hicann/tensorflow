# reduce_scatter

## 功能说明

集合通信算子ReduceScatter的操作接口，将通信域内所有rank的输入数据均分成rank size份，然后分别取每个rank的rank size之一份数据进行归约操作（如sum、prod、max、min）。最后，将结果按照编号分散到各个rank的输出buffer。

![](../figures/reduce_scatter.png)

## 函数原型

```python
def reduce_scatter(tensor, reduction, rank_size, group="hccl_world_group", fusion=0, fusion_id=-1)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | TensorFlow的tensor类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、int64、uint64、float16、float32、float64、bfp16。针对int64、uint64、float64，当前仅支持节点内通信。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。<br>针对Atlas 训练系列产品，支持数据类型：int8、int32、int64、float16、float32。<br>针对Atlas 300I Duo 推理卡，支持数据类型：int8、int16、int32、float16、float32。<br>需要注意tensor的第一个维度的元素个数必须是rank size的整数倍。 |
| reduction | 输入 | 归约操作类型，String类型。<br>针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、max、min。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的操作类型为sum、max、min、prod，当前版本“prod”操作不支持int16、bfp16数据类型。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的操作类型为sum、max、min、prod，当前版本“prod”操作不支持int16、bfp16数据类型。<br>针对Atlas 300I Duo 推理卡，支持的操作类型为sum、max、min、prod，当前版本“max”、“min”、“prod”操作不支持int16数据类型。 |
| rank_size | 输入 | group内device的数量，int类型。<br>最大值：32768。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |
| fusion | 输入 | reducescatter算子融合标识，int类型，支持以下取值：<br><br>  - 0：网络编译时，不会对该算子进行融合，即该reducescatter算子不和其他reducescatter算子融合。<br>  - 2：网络编译时，会对reducescatter算子按照相同的fusion_id进行融合，即“fusion_id”相同的reducescatter算子之间会进行融合。 |
| fusion_id | 输入 | reducescatter算子的融合id，int类型。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的reducescatter算子进行融合。 |

## 返回值

对输入tensor执行完reducescatter操作之后的结果tensor。

## 约束说明

- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 该接口要求输入tensor的数据量不超过8GB。
- reducescatter算子融合场景只支持reduction操作类型sum。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((2, 3), minval=1, maxval=10, dtype=tf.float32)
rank_size = 2
result = hccl_ops.reduce_scatter(tensor, "sum", rank_size)
```
