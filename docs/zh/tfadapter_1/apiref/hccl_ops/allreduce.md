# allreduce

## 功能说明

集合通信算子AllReduce的操作接口，将group内所有节点的输入数据进行归约操作后，再把结果发送到所有节点的输出buf，其中归约操作类型由reduction参数指定。

![](../figures/allreduce.png)

## 函数原型

```python
def allreduce(tensor, reduction, fusion=1, fusion_id=-1, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | TensorFlow的tensor类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、int64、uint64、float16、float32、float64、bfp16。针对int64、uint64、float64，当前仅支持节点内通信。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。<br>针对Atlas 训练系列产品，支持数据类型：int8、int32、int64、float16、float32。<br>针对Atlas 300I Duo 推理卡，支持数据类型：int8、int16、int32、float16、float32。|
| reduction | 输入 | 归约操作类型，String类型。<br>针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、max、min。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的操作类型为sum、max、min、prod，当前版本“prod”操作不支持int16、bfp16数据类型。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的操作类型为sum、max、min、prod，当前版本“prod”操作不支持int16、bfp16数据类型。<br>针对Atlas 300I Duo 推理卡，支持的操作类型为sum、max、min、prod，当前版本“max”、“min”、“prod”操作不支持int16数据类型。|
| fusion | 输入 | allreduce算子融合标识，int类型，支持以下取值：<br><br>  - 0：网络编译时，不会对该算子进行融合，即该allreduce算子不和其他allreduce算子融合。<br>  - 1：网络编译时，对该算子按照梯度切分策略进行融合。<br>  - 2：网络编译时，对allreduce算子按照相同的fusion_id进行融合，即“fusion_id”相同的allreduce算子之间会进行融合。|
| fusion_id | 输入 | allreduce算子的融合id，int类型。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的allreduce的算子进行融合。|
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。|

## 返回值

对输入tensor执行完allreduce操作之后的结果tensor。

## 约束说明

- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 每个rank只能有一个输入。
- allreduce上游节点暂不支持variable算子。
- 该接口要求输入tensor的数据量不超过8GB。
- allreduce算子融合场景只支持reduction操作类型sum。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
result = hccl_ops.allreduce(tensor, "sum")
```
