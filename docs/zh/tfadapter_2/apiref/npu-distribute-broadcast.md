# npu.distribute.broadcast

## 功能说明

集合通信算子Broadcast的操作接口，用于NPU分布式部署场景下，worker间的变量同步。

该接口可配合HCCL提供的Python语言的通信域管理接口进行使用，关于HCCL Python接口介绍可参见《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“API  \> 通信域管理 \> Python语言接口”。

## 函数原型

```python
npu.distribute.broadcast(values, root_rank, fusion=2, fusion_id=0, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| values | 输入 | 单个TensorFlow的Variable或者Variable的集合。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，tensor支持的数据类型为int8、int32、float16、float32、int64、uint64、bfloat16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，tensor支持的数据类型为int8、int32、float16、float32、int64、uint64、bfloat16。<br>针对Atlas 训练系列产品，tensor支持的数据类型为int8、int32、float16、float32、int64、uint64。 |
| root_rank | 输入 | int类型。<br>作为root节点的rank_id，该id是group内的rank id。 |
| fusion | 输入 | int类型。<br>broadcast算子融合标识，支持以下取值：<br>  - 0：标识网络编译时，不会对该算子进行融合，即该broadcast算子不和其他broadcast算子融合。<br>  - 2：网络编译时，会对broadcast算子按照相同的fusion_id进行融合，即“fusion_id”相同的broadcast算子之间会进行融合。 |
| fusion_id | 输入 | int类型。<br>broadcast算子的融合id。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的broadcast算子进行融合。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

无。

## 调用示例

将0卡上的变量广播到其他卡：

```python
# rank_id = 0  rank_size = 8
import npu_device as npu
x = tf.Variable(tf.random.normal(shape=()))
print("before broadcast", x)
npu.distribute.broadcast(x, root_rank=0)
print("after_broadcast", x)
```

广播前：

![](figures/before_broadcast.png)

广播后：

![](figures/after_broadcast.png)
