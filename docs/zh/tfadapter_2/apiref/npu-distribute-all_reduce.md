# npu.distribute.all_reduce

## 功能说明

集合通信算子AllReduce的操作接口，用于NPU分布式部署场景下，worker间的聚合运算。

该接口可配合HCCL提供的Python语言的通信域管理接口进行使用，关于HCCL Python接口介绍可参见《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“API \> 通信域管理 \> Python语言接口”。

## 函数原型

```python
npu.distribute.all_reduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| values | 输入 | TensorFlow的tensor类型。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，tensor支持的数据类型为int8、int32、float16、float32、bfloat16（prod操作不支持）。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，tensor支持的数据类型为int8、int32、float16、float32、bfloat16（prod操作不支持）。<br>针对Atlas 训练系列产品，tensor支持的数据类型为int8、int32、float16、float32。|
| reduction | 输入 | String类型。<br>聚合运算的类型，可以为"mean"、"max"、"min"、"prod"或"sum"。 |
| fusion | 输入 | int类型。<br>allreduce算子融合标识，支持以下取值：<br>  - 0：网络编译时，不会对该算子进行融合，即该allreduce算子不和其他allreduce算子融合。<br>  - 1：网络编译时，对该算子按照梯度切分策略进行融合。<br>  - 2：网络编译时，对allreduce算子按照相同的fusion_id进行融合，即“fusion_id”相同的allreduce算子之间会进行融合。 |
| fusion_id | 输入 | int类型。<br>allreduce算子的融合id。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的allreduce算子进行融合。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

对values进行聚合运算后的结果，类型与values一致，值与values输入一一对应。

## 调用示例

在多卡上聚合某个值：

```python
# rank_id = 0  rank_size = 8
import npu_device as npu
v = tf.constant(1.0)
x = npu.distribute.all_reduce([v], 'sum') # 8.0
y = npu.distribute.all_reduce([v], 'mean') # 1.0
```
