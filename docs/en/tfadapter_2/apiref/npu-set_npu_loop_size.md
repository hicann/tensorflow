# npu.set_npu_loop_size

## Description

Sets the number of iterations \(or steps\) per loop offloaded to the NPU.

## Prototype

```python
npu.set_npu_loop_size(loop_size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| loop_size | Input | Number of steps per loop. Must be a positive integer. |

## Returns

None

## Example

```python
import npu_device as npu
npu.set_npu_loop_size(100) # Set 100 steps to offload.
```
