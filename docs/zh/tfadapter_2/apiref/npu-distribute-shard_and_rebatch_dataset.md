# npu.distribute.shard_and_rebatch_dataset

## 功能说明

用于NPU分布式部署场景下，不同worker上数据集分片及batch大小调整。

## 函数原型

```python
npu.distribute.shard_and_rebatch_dataset(dataset, global_bs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dataset | 输入 | TensorFlow的Dataset类型。<br>需要进行切分的数据集。 |
| global_bs | 输入 | 全局batch的大小。 |

## 返回值

返回一个2个元素的tuple对象，第一个元素为切分后的Dataset，第二个元素为每个worker应当处理的实际batch大小。

## 调用示例

```python
import npu_device as npu
dataset, batch_size = npu.distribute.shard_and_rebatch_dataset(dataset, batch_size)
```
