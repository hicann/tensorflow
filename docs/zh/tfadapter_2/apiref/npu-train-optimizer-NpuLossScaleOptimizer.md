# npu.train.optimizer.NpuLossScaleOptimizer

## 功能说明

NPU提供的LossScaleOptimizer，浮点计算的溢出模式为“饱和模式”的场景下，由于NPU上的溢出运算不保证输出Inf或者NaN，所以此种场景下，使用LossScaleOptimizer的脚本应当替换为该优化器，来屏蔽溢出检测的差异。

- Atlas 训练系列产品，浮点计算的溢出模式默认为“饱和模式”，且仅支持“饱和模式”。饱和模式是指当计算出现溢出时，饱和为浮点数极值（+-MAX）。
- 其他系列产品，浮点计算支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

## 函数原型

```python
npu.train.optimizer.NpuLossScaleOptimizer(inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None)
```

## 参数说明

该优化器继承自tf.keras.mixed_precision.LossScaleOptimizer，使用方式完全相同，可以参考[LINK](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer)。

## 返回值

NpuLossScaleOptimizer类型优化器。

## 调用示例

```python
import npu_device as npu
optimizer = tf.keras.optimizers.Adam(lr=0.1)
optimizer = npu.train.optimizer.NpuLossScaleOptimizer(optimizer)
```
