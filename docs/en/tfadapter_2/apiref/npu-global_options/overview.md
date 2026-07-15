# Overview

## Description

Returns a global singleton configuration object for initializing an NPU device. By modifying the parameters of the global singleton object, you can control the initialization options of the NPU device. This API must be called before the  **npu.open**  API call.

This section describes the global singleton configuration provided by the NPU.

## Prototype

npu.global_options\(\)

## Example

Set the global configuration options before initializing the NPU. The following is a call example of changing the precision mode from  **allow_fp32_to_fp16**  to  **allow_mix_precision**:

```python
import npu_device as npu
npu.global_options().precision_mode = 'allow_mix_precision'
npu.open().as_default()
```
