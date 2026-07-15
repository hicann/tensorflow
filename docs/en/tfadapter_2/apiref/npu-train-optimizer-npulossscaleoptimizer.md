# npu.train.optimizer.NpuLossScaleOptimizer

## Description

When the overflow/underflow mode of floating-point computation is saturation mode, the overflow/underflow computation on the NPU may not output  **Inf**  or  **NaN**. Therefore, you should replace LossScaleOptimizer in the script with NpuLossScaleOptimizer, to mask the differences in overflow/underflow detection.

## Prototype

```python
npu.train.optimizer.NpuLossScaleOptimizer(inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None)
```

## Parameters

The optimizer is inherited from tf.keras.mixed_precision.LossScaleOptimizer and works in the same way. For details, see  [LINK](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer).

## Returns

NpuLossScaleOptimizer optimizer.

## Example

```python
import npu_device as npu
optimizer = tf.keras.optimizers.Adam(lr=0.1)
optimizer = npu.train.optimizer.NpuLossScaleOptimizer(optimizer)
```
