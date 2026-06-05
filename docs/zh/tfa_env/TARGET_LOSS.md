# TARGET_LOSS

## 功能描述

TensorFlow  1.15训练场景下，若通过“experimental_accelerate_train_mode”参数或者“accelerate_train_mode”参数触发了训练加速功能，可通过此环境变量设置NPU上的目标训练loss值。

- 若训练脚本中使用session配置，关于该环境变量使用场景的详细介绍可参见session配置中的[experimental_accelerate_train_mode](../tfadapter_1/apiref/session_config/experiment_options.md#experimental_accelerate_train_mode)参数的介绍。
- 若训练脚本中使用NPURunConfig配置，关于该环境变量使用场景的详细介绍可参见npu.npu_config中ExperimentalConfig构造函数的[accelerate_train_mode](../tfadapter_1/apiref/npu_config/experimentalconfig_constructor.md)参数的介绍。

该环境变量取值为浮点类型，无默认值。

## 配置示例

```bash
export TARGET_LOSS=3.0
```

## 使用约束

该环境变量仅适用于TensorFlow  1.15网络在昇腾平台执行训练的场景。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 训练系列产品
