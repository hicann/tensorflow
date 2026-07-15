# NPULossScaleOptimizer Constructor

## Description

Constructor of the  **NPULossScaleOptimizer**  class, which is used to enable loss scaling in mixed precision training when the overflow/underflow mode of floating-point computation is saturation mode. Loss scaling solves the underflow problem caused by the small float16 representation range. The  **NPULossScaleOptimizer**  class inherits the  **LossScaleOptimizer**  class and can call native APIs of the base class.

- For the  Ascend 950PR/Ascend 950DT, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas A3 training product/Atlas A3 inference product, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas A2 training product/Atlas A2 inference product, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas training product, the default overflow/underflow mode of floating-point computation is saturation, and only the saturation mode is supported. This means when an overflow/underflow occurs during computation, the computation result is saturated to a floating-point extreme value \(±MAX\).

## Prototype

```python
class NPULossScaleOptimizer(lso.LossScaleOptimizer):
    def __init__(self, opt, loss_scale_manager, is_distributed=False)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| opt | Input | Single-server training optimizer for gradient calculation and weight update. |
| loss_scale_manager | Input | Loss scaling update mode, including static update and dynamic update.<br><br>  - Before creating NPULossScaleOptimizer, you can instantiate a FixedLossScaleManager class to set the loss scaling with a static value. For details about the constructor of the FixedLossScaleManager class, see [FixedLossScaleManager Constructor](../npu_loss_scale_manager/FixedLossScaleManager_constructor.md).<br>  - Before creating NPULossScaleOptimizer, you can instantiate an ExponentialUpdateLossScaleManager class to dynamically configure loss scaling. For details about the constructor of the ExponentialUpdateLossScaleManager class, see [ExponentialUpdateLossScaleManager Constructor](../npu_loss_scale_manager/ExponentialUpdateLossScaleManager_constructor.md). |
| is_distributed | Input | Used to support the loss scaling function in the distributed training scenario.<br><br>  - True: Set this parameter to True for distributed training.<br>  - False |

## Returns

An object of the  **NPULossScaleOptimizer**  class

## Example

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
