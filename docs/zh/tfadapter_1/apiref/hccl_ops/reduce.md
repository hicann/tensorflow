# reduce

## 功能说明

集合通信算子Reduce的操作接口，将所有rank的数据相加（或其他归约操作）后，再把结果发送到root节点的指定位置上。

![](../figures/reduce.png)

## 函数原型

```python
def reduce(tensor, reduction, root_rank, fusion=0, fusion_id=-1, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | TensorFlow的tensor类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、int64、uint64、float16、float32、float64、bfp16。针对int64、uint64、float64，当前仅支持节点内通信。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。<br>针对Atlas 训练系列产品，支持数据类型：int8、int32、int64、float16、float32。 |
| reduction | 输入 | 归约操作类型，String类型。<br>针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、max、min。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的操作类型为sum、max、min、prod，当前版本“prod”操作不支持int16、bfp16数据类型。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的操作类型为sum、max、min、prod，当前版本“prod”操作不支持int16、bfp16数据类型。 |
| root_rank | 输入 | 作为root节点的rank_id，该id是group内的rank id，int类型。 |
| fusion | 输入 | reduce算子融合标识，int类型，支持以下取值：<br><br>  - 0：不融合，该reduce算子不和其他reduce算子融合。<br>  - 2：按照相同fusion_id进行融合。 |
| fusion_id | 输入 | reduce算子的融合id，int类型。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的reduce算子进行融合。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

对输入tensor执行完reduce操作之后的结果tensor。

## 约束说明

- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 该接口要求输入tensor的数据量不超过8GB。
- reduce算子融合场景只支持reduction操作类型sum。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
result = hccl_ops.reduce(tensor, "sum", 0)
```
