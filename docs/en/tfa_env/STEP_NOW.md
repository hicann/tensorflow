# STEP_NOW

## Description

In the  TensorFlow  1.15 training scenario, if the training acceleration function is enabled through the  **experimental_accelerate_train_mode**  or  **accelerate_train_mode**  parameter, you can use this environment variable to set the number of execution steps on the NPU.

- If the session configuration is used in the training script, see the description of the  [experimental_accelerate_train_mode](../tfadapter_1/apiref/session_config/experiment_options.md#experimental_accelerate_train_mode) parameter.
- If  **NPURunConfig**  is used in the training script, see the description of the [accelerate_train_mode](../tfadapter_1/apiref/npu_config/experimentalconfig_constructor.md) parameter.

The value of this environment variable is of the int type. There is no default value.

## Example

```bash
export STEP_NOW=100
```

## Constraints

This environment variable applies only to the scenario where the  TensorFlow  1.15 network is trained on the Ascend platform.

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas training product
