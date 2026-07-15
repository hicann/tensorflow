# ASCEND_DEVICE_ID

## Description

Specifies the logical ID of the AI processor used by the current process.

The value range is \[0, N – 1\] and the default value is 0. N indicates the number of devices on the physical machine, VM, or in a container.

Use cases:

The TensorFlow framework network performs training or online inference on the Ascend platform.

## Example

```bash
export ASCEND_DEVICE_ID=0
```

## Constraints

无

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas inference product

Atlas training product
