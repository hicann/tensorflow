# broadcast

## 功能说明

集合通信算子Broadcast的操作接口，将通信域内root节点的数据广播到其他rank。

![](../figures/broadcast.png)

## 函数原型

```python
def broadcast(tensor, root_rank, fusion=2,fusion_id=0, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | TensorFlow的tensor类型，list类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。<br>针对Atlas 300I Duo 推理卡，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。 |
| root_rank | 输入 | 作为root节点的rank_id，该id是group内的rank id，int类型。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |
| fusion | 输入 | broadcast算子融合标识，int类型，支持以下取值：<br><br>  - 0：标识网络编译时，不会对该算子进行融合，即该broadcast算子不和其他broadcast算子融合。<br>  - 2：网络编译时，会对broadcast算子按照相同的fusion_id进行融合，即“fusion_id”相同的broadcast算子之间会进行融合。 |
| fusion_id | 输入 | broadcast算子的融合id，int类型。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的broadcast算子进行融合。 |

## 返回值

对输入tensor执行完broadcast操作之后的结果tensor。

## 约束说明

- 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
- 如果两个Broadcast算子有输入输出的依赖关系，则不能对其进行融合，否则可能会出现图成环问题。

    如下图所示，broadcast2与broadcast1之间存在输入输出依赖关系，所以不能对broadcast1与broadcast2算子进行融合，即调用此“broadcast”接口时，“fusion”参数需要设置为“0”。

    ![](../figures/broadcast_restrict.png)

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
inputs = [tensor]
root = 0
result = hccl_ops.broadcast(inputs, root)
```
