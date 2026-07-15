# ExponentialUpdateLossScaleManager Constructor

## Description

Constructor of the  **ExponentialUpdateLossScaleManager**  class, which is used to define the dynamic  **LossScale**  parameter during training and dynamically obtains and updates the value of  **LossScale**  by defining the  **loss_scale**  variable when the overflow/underflow mode of floating-point computation is saturation mode.

- For the  Ascend 950PR/Ascend 950DT, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas A3 training product/Atlas A3 inference product, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas A2 training product/Atlas A2 inference product, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.
- For the  Atlas training product, the default overflow/underflow mode of floating-point computation is saturation, and only the saturation mode is supported. This means when an overflow/underflow occurs during computation, the computation result is saturated to a floating-point extreme value \(±MAX\).

## Prototype

```python
class ExponentialUpdateLossScaleManager(lsm_lib.ExponentialUpdateLossScaleManager):
    def __init__(self,
                 init_loss_scale,
                 incr_every_n_steps,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2,
                 decr_ratio=0.8)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| init_loss_scale | Input | Initial loss scale value. A float. |
| incr_every_n_steps | Input | If no overflow occurs for N iterations, increase the value of loss scale. |
| decr_every_n_nan_or_inf | Input | If an overflow occurs for N iterations, decrease the value of loss scale. Defaults to 2. |
| incr_ratio | Input | Percentage increase of loss scale. Defaults to 2. |
| decr_ratio | Input | Percentage decrease of loss scale. Defaults to 0.8. |

## Returns

An object of the  **ExponentialUpdateLossScaleManager**  class

## Restrictions

The objects of the  **ExponentialUpdateLossScaleManager**  class cannot be constructed within the influence range of  **tf.control_dependencies\(\)**. Otherwise, the graph structure execution sequence may be different from the expected sequence.
