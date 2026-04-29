# NPULossScaleOptimizer构造函数

## 功能说明

NPULossScaleOptimizer类的构造函数，浮点计算的溢出模式为“饱和模式”的场景下，用于在混合精度训练中使能Loss Scaling。Loss Scaling解决了由于float16表示范围较小导致的下溢出问题。NPULossScaleOptimizer类继承了LossScaleOptimizer类，可以调用基类的原生接口。

- Atlas 训练系列产品，浮点计算的溢出模式默认为“饱和模式”，且仅支持“饱和模式”。饱和模式是指当计算出现溢出时，饱和为浮点数极值（+-MAX）。
- 其他系列产品，浮点计算支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

## 函数原型

```python
class NPULossScaleOptimizer(lso.LossScaleOptimizer):
    def __init__(self, opt, loss_scale_manager, is_distributed=False)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opt | 输入 | 用于梯度计算和更新权重的单机版训练优化器。 |
| loss_scale_manager | 输入 | 用于决定LossScale的更新方式，包括静态更新和动态更新。<br>  - 用户在创建NPULossScaleOptimizer之前，可以实例化一个FixedLossScaleManager类进行静态LossScale的配置（LossScale值固定，用户需指定LossScale值）。FixedLossScaleManager类的构造函数，请参见[FixedLossScaleManager构造函数](FixedLossScaleManager_constructor.md)。<br>  - 用户在创建NPULossScaleOptimizer之前，可以实例化一个ExponentialUpdateLossScaleManager类进行动态LossScale的配置。ExponentialUpdateLossScaleManager类的构造函数，请参见[ExponentialUpdateLossScaleManager构造函数](ExponentialUpdateLossScaleManager_constructor.md)。 |
| is_distributed | 输入 | 用于支持分布式训练场景的Loss Scaling功能。取值：<br> - True：分布式训练时需要配置为True。<br>  - False。 |

## 返回值

返回NPULossScaleOptimizer类对象。

## 调用示例

```python
from npu_bridge.npu_init import *

if FLAGS.use_fp16 and (FLAGS.npu_bert_loss_scale not in [None, -1]):
  opt_tmp = opt
  if FLAGS.npu_bert_loss_scale == 0:
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
  elif FLAGS.npu_bert_loss_scale >= 1:
    loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.npu_bert_loss_scale)
  else:
    raise ValueError("Invalid loss scale: %d" % FLAGS.npu_bert_loss_scale)
  if ops_adapter.size() > 1:
    opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager, is_distributed=True)
  else:
    opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
```
