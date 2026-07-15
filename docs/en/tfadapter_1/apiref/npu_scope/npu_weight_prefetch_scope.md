# npu_weight_prefetch_scope

## Description

Identifies the operators whose weight data will be prefetched into a buffer pool and specifies the ID and size of the buffer pool.

A prefetch buffer pool is an independent area of  AI processor  memory. The size is determined before compilation, based on which prefetch tasks are controlled. If a buffer pool is full, prefetch tasks reuse the memory from the start address of the pool with timing control.

For an ultra-large model trained on a cluster, if weights are distributed to  AI processors, only 1/_N_  \(_N_  indicates the number of  AI processors participating in training\) weight data is stored on each device, reducing the memory footprint of the large model on each device. Before the compute operators are executed, the full weight data needs to be pulled to the local host. To avoid lack of memory, read-ahead weight data is stored in buffer pools.

## Prototype

```python
def npu_weight_prefetch_scope(buffer_pool_id=0, buffer_pool_size=536870912)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| buffer_pool_id | Input | An int, indicating the ID of the buffer pool to enable. Defaults to 0. |
| buffer_pool_size | Input | Size (bytes) of the specified buffer pool. Defaults to 536870912 (about 512 MB). |

## Returns

None

## Restrictions

1. The prefetch buffer pool supports only prefetch operators with single input and single output.
2. The sizes of buffer pools with the same ID must be the same.
3. The buffer pool must be large enough for the largest prefetch operator, including its possible aligned and padded parts.
4. The prefetch buffer pool is not supported for prefetch operators in a subgraph or control flow.

## Example

```python
from npu_bridge.estimator.npu.npu_scope import npu_weight_prefetch_scope

 ... ...

with npu_weight_prefetch_scope():
    # The output memory of AllGather uses the default buffer pool.
    global_weight1 = hcom.allgather(local_weight1)

 ... ...

with npu_weight_prefetch_scope(1, 268435456): # 256 MB: 256 x 1024 x 1024
    # The output memory of AllGather uses the 256 MB buffer pool indexed 1.
    global_weight2 = hcom.allgather(local_weight2)
```
