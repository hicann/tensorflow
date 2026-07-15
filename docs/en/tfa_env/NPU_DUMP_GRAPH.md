# NPU_DUMP_GRAPH

## Description

Enables or disables graph dump of the TF Adapter in the  TensorFlow  2.6.5 training and online inference scenario.

- **1** or **true**: enabled.
- **0** or **false**: disabled.

## Example

```bash
export NPU_DUMP_GRAPH=1
```

## Constraints

- This environment variable should be set before the  **import npu_device**  operation.
- This environment variable applies only to the scenario where the  TensorFlow  2.6.5 network training or online inference is performed on the Ascend platform.

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas training product

Atlas inference product
