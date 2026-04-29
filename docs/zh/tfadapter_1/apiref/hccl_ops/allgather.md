# allgather

## 功能说明

集合通信算子AllGather的操作接口，将通信域内所有节点的输入按照rank id重新排序，然后拼接起来，再将结果发送到所有节点的输出。

![](../figures/allgather.png)

> [!NOTE]说明
> 针对AllGather操作，每个节点都接收按照rank id重新排序后的数据集合，即每个节点的AllGather输出都是一样的。

## 函数原型

```python
def allgather(tensor, rank_size, group="hccl_world_group", fusion=0, fusion_id=-1)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | TensorFlow的tensor类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。<br>针对Atlas 300I Duo 推理卡，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。 |
| rank_size | 输入 | group内device的数量，int类型。<br>最大值为32768。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |
| fusion | 输入 | AllGather算子融合标识，int类型，支持以下取值：<br><br>  - 0：标识网络编译时，不会对该算子进行融合，即该AllGather算子不和其他AllGather算子融合。<br>  - 2：网络编译时，会对AllGather算子按照相同的fusion_id进行融合，即“fusion_id”相同的AllGather算子之间会进行融合。 |
| fusion_id | 输入 | AllGather算子的融合id，int类型。<br>当“fusion”取值为“2”时，网络编译时对相同fusion_id的AllGather算子进行融合。 |

## 返回值

对输入tensor执行完allgather操作之后的结果tensor。

## 约束说明

调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
rank_size = 2
result = hccl_ops.allgather(tensor, rank_size)
```
