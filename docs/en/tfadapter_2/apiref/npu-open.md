# npu.open

## Description

Registers an NPU device, used in conjunction with  **as_default**  to set the NPU as the default device.

## Prototype

```python
npu.open(device_id=None).as_default()
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| device_id | Input | ID of the NPU device to be initialized. Defaults to the value of the ASCEND_DEVICE_ID environment variable or 0 if the ASCEND_DEVICE_ID environment variable is not set.<br>ASCEND_DEVICE_ID specifies the logical ID of the AI processor used by the current process. Value range: [0, N-1]. Example:<br>`export ASCEND_DEVICE_ID=0` |

## Returns

A  **npu_device**  instance.

## Example

```python
import npu_device as npu
npu.open().as_default()
```
