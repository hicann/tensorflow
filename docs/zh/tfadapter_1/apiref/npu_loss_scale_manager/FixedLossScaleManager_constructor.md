# FixedLossScaleManager构造函数

## 功能说明

FixedLossScaleManager类的构造函数，浮点计算的溢出模式为“饱和模式”的场景下，可通过此接口定义训练场景下的静态LossScale参数。

- Atlas 训练系列产品，浮点计算的溢出模式默认为“饱和模式”，且仅支持“饱和模式”。饱和模式是指当计算出现溢出时，饱和为浮点数极值（+-MAX）。
- 其他系列产品，浮点计算支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

## 函数原型

```python
class FixedLossScaleManager(lsm_lib.FixedLossScaleManager):
    def __init__(self, loss_scale, enable_overflow_check=True)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| loss_scale | 输入 | LossScale值。float类型，取值不能小于1。<br>选择过小的LossScale的值可能会影响模型收敛，选择较大的LossScale可能会导致训练过程出现溢出。可以与GPU的值保持一致。 |
| enable_overflow_check | 输入 | 参数更新时，是否检查溢出。<br>  - True：检测到有溢出的迭代，会放弃参数更新，默认是True。<br>  - False：始终更新参数，不检查迭代中是否出现溢出。 |

## 返回值

返回FixedLossScaleManager类对象。
