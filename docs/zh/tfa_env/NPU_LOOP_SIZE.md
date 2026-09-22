# NPU_LOOP_SIZE

## 功能描述

TensorFlow  2.6.5训练与在线推理场景下，用于设置NPU上循环下沉的次数。

## 配置示例

```bash
export NPU_LOOP_SIZE=32
```

## 使用约束

- 该变量需要在import npu_device前设置。
- 该环境变量仅适用于TensorFlow  2.6.5网络在昇腾平台执行训练或在线推理的场景。

## 支持的型号

Ascend 950PR&950DT系列产品

Atlas A3系列产品

Atlas A2系列产品

Atlas推理系列产品

Atlas训练系列产品
