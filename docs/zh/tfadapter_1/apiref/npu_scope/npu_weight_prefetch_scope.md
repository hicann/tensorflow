# npu_weight_prefetch_scope

## 功能说明

用于标识哪些算子使用权重预取缓存池内存，并指定使用的缓存池的id以及大小。

预取缓存池内存是在AI处理器上分配出来的一块独立的内存区域，大小在编译前确定，预取任务按照缓存池大小进行控制，当缓存池已满，则重新从池的起始位置复用内存，复用内存的任务之间有时序控制。

对于超大模型的集群场景下，如果使能weight分布式存储优化，每个AI处理器上仅需存储1/N（N为参与训练的AI处理器数）weight数据量，从而降低整个大模型对内存的占用量。 在每层计算类算子启动前，需要把全量weight数据拉取到本地。为了缓解内存资源紧张，weight的预取操作使用缓存池内存。

## 函数原型

```python
def npu_weight_prefetch_scope(buffer_pool_id=0, buffer_pool_size=536870912)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| buffer_pool_id | 输入 | 指定使能的缓存池ID，取值为整数。默认值为0。 |
| buffer_pool_size | 输入 | 指定该ID的缓存池大小，单位是bytes，默认值为536870912，约为512MB 。 |

## 返回值

无

## 约束说明

1. 预取缓存池只支持单输入单输出的预取算子；
2. 相同ID的缓存池设置的大小必须一致；
3. 缓存池的大小必须满足占用内存最大的预取算子的内存需求，包括对齐及填充的内容；
4. 不支持预取算子位于子图或者控制流分支的情况配置使用预取缓存池。

## 调用示例

```python
from npu_bridge.estimator.npu.npu_scope import npu_weight_prefetch_scope

 ... ...

with npu_weight_prefetch_scope():
    # allgather的输出内存使用默认设置的缓存池
    global_weight1 = hcom.allgather(local_weight1)

 ... ...

with npu_weight_prefetch_scope(1, 268435456): # 256MB: 256 * 1024 * 1024
    # allgather的输出内存使用id为1，大小为256MB的缓存池
    global_weight2 = hcom.allgather(local_weight2)
```
