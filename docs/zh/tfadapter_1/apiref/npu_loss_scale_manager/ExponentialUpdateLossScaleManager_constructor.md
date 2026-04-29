# ExponentialUpdateLossScaleManager构造函数

## 功能说明

ExponentialUpdateLossScaleManager类的构造函数，浮点计算的溢出模式为“饱和模式”的场景下，用于定义训练场景下的动态LossScale参数，并通过定义loss_scale变量动态获取和更新LossScale值。

- Atlas 训练系列产品，浮点计算的溢出模式默认为“饱和模式”，且仅支持“饱和模式”。饱和模式是指当计算出现溢出时，饱和为浮点数极值（+-MAX）。
- 其他系列产品，浮点计算支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

## 函数原型

```python
class ExponentialUpdateLossScaleManager(lsm_lib.ExponentialUpdateLossScaleManager):
    def __init__(self,
                 init_loss_scale,
                 incr_every_n_steps,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2,
                 decr_ratio=0.8)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| init_loss_scale | 输入 | 初始LossScale值。float类型。 |
| incr_every_n_steps | 输入 | 当累计N次迭代未出现溢出时，增大LossScale值。 |
| decr_every_n_nan_or_inf | 输入 | 当累计N次迭代出现溢出时，减小LossScale值。默认值：2。 |
| incr_ratio | 输入 | LossScale增大的比例。默认值：2。 |
| decr_ratio | 输入 | LossScale减小的比例。默认值：0.8。 |

## 返回值

返回ExponentialUpdateLossScaleManager类对象。

## 约束说明

ExponentialUpdateLossScaleManager类对象的构造不能在tf.control_dependencies\(\)接口的作用域内，否则可能会造成图结构执行顺序与预期不一致。
