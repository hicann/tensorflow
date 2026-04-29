# npu_allreduce

## 功能说明

梯度计算完成后，对梯度进行allreduce和梯度更新。

## 函数原型

```python
def _npu_allreduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| values | 输入 | tensor的list或者tensor。 |
| reduction | 输入 | reduce的op类型，可以为“sum”或“mean”。 |
| fusion | 输入 | int类型，算子融合标识。<br>  - 0：不融合，该allreduce算子不和其他allreduce算子融合。<br>  - 1（默认值）：按照梯度切分策略进行融合。<br>  - 2：按照相同fusion_id进行融合。 |
| fusion_id | 输入 | 算子融合索引标志，对相同fusion_id的allreduce算子进行融合。 |
| group | 输入 | String类型，group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

返回list tensor或者tensor，和输入类型保持一致。

## 调用示例

```python
from npu_bridge.npu_init import *
grads = npu_allreduce(tf.gradients(a + b, [a, b], stop_gradients=[a, b]))
```
