# NPU_LOOP_SIZE

## Description

Sets the number of iterations per loop offloaded to the NPU in the  TensorFlow  2.6.5 training and online inference scenarios.

## Example

```bash
export NPU_LOOP_SIZE=32
```

## Constraints

- This variable should be set before the  **import npu_device**  operation.
- This environment variable applies only to the scenario where the  TensorFlow  2.6.5 network training or online inference is performed on the Ascend platform.

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas training product

Atlas inference product
