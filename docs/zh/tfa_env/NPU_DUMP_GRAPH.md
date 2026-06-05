# NPU_DUMP_GRAPH

## 功能描述

TensorFlow  2.6.5训练与在线推理场景下，用于开启TF Adapter图Dump功能。

- "1"或"true"：开启图Dump功能
- "0"或"false"：关闭图Dump功能

## 配置示例

```bash
export NPU_DUMP_GRAPH=1
```

## 使用约束

- 该环境变量需要在import npu_device前设置。
- 该环境变量仅适用于TensorFlow  2.6.5网络在昇腾平台执行训练或在线推理的场景。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 推理系列产品

Atlas 训练系列产品
