# set_device_sat_mode

## Description

Sets the process-level overflow/underflow mode for floating-point computation.

- Saturation mode: When overflow occurs during compute, the compute result is saturated as the floating-point extremum \(+-MAX\).
- INF/NaN mode: Complies with IEEE 754 and outputs the INF/NaN compute result based on the definition.

For the  Ascend 950PR/Ascend 950DT, the overflow/underflow mode can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.

For the  Atlas A3 training product/Atlas A3 inference product, the overflow/underflow mode can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.

For the  Atlas A2 training product/Atlas A2 inference product, the overflow/underflow mode can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.

For the  Atlas training product, the default \(and the only supported\) mode is saturation mode.

## Prototype

```python
def set_device_sat_mode(mode)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| mode | Input | Specified overflow mode.<br><br>  - 0: saturation mode.<br>  - 1: INF/NaN mode.<br><br>For the Ascend 950PR/Ascend 950DT, use the default value 1.<br>For the Atlas A3 training product/Atlas A3 inference product, use the default value 1.<br>For the Atlas A2 training product/Atlas A2 inference product, use the default value 1.<br>For the Atlas training product, the default (and the only supported) value is 0. |

## Returns

None

## Restrictions

This API needs to be configured during running and called before the network script is executed.

## Example

The following example applies only to the  Ascend 950PR/Ascend 950DTAtlas A3 training product/Atlas A3 inference productAtlas A2 training product/Atlas A2 inference product. For other processors, you do not need to explicitly call this API.

```python
import tensorflow as tf
from npu_bridge.npu_init import *

......
# The following API is called to set the overflow/underflow mode during network execution:
npu_plugin.set_device_sat_mode(1)
sess.run(xxx)

```
