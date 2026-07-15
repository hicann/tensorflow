# npu.distribute.shard_and_rebatch_dataset

## Description

Shards the dataset and global batch size for workers in distributed NPU training.

## Prototype

```python
npu.distribute.shard_and_rebatch_dataset(dataset, global_bs)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| dataset | Input | TensorFlow dataset type.<br>Dataset to be sharded. |
| global_bs | Input | Global batch size. |

## Returns

A tuple object of two elements, for the sharded datasets and the per-worker mini-batch size respectively.

## Example

```python
import npu_device as npu
dataset, batch_size = npu.distribute.shard_and_rebatch_dataset(dataset, batch_size)
```
