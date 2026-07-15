# FixedLossScaleManager Constructor

## Description

Constructor of the  **FixedLossScaleManager**  class, which is used to define the static  **LossScale**  parameter during training when the overflow/underflow mode of floating-point computation is saturation mode.

- For the  Ascend 950PR/Ascend 950DT, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas A3 training product/Atlas A3 inference product, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas A2 training product/Atlas A2 inference product, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas training product, the default overflow/underflow mode of floating-point computation is saturation, and only the saturation mode is supported. This means when an overflow/underflow occurs during computation, the computation result is saturated to a floating-point extreme value \(±MAX\).

## Prototype

```python
class FixedLossScaleManager(lsm_lib.FixedLossScaleManager):
    def __init__(self, loss_scale, enable_overflow_check=True)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| loss_scale | Input | Loss scale value. The value is of the float type and cannot be less than 1.<br>If the value of loss scale is too small, model convergence may be affected. If the value of loss scale is too large, overflow may occur during training. The value can be the same as that of GPU. |
| enable_overflow_check | Input | Overflow detection enable during parameter update.<br>  - True (default): enabled. If overflow is detected in an iteration, parameters of that iteration are not updated.<br>  - False: disabled. Parameters are always updated regardless of overflow. |

## Returns

An object of the  **FixedLossScaleManager**  class
