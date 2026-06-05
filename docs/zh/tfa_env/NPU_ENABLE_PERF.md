# NPU_ENABLE_PERF

## 功能描述

TensorFlow  2.6.5训练与在线推理场景下，用于开启TF Adapter图耗时打印功能。

- "1"或"true"：开启图耗时打印功能
- "0"或"false"：关闭图耗时打印功能

开启TF Adapter图耗时打印功能后，开发者可搜索日志中的“Graph engine run”与“cost”关键字查询相关耗时，例如：

```text
Graph engine run 1 times for graph 0 cost 11257 ms
```

## 配置示例

```bash
export NPU_ENABLE_PERF=1
```

## 使用约束

- 该环境变量仅在开启TF Adapter的Debug级别日志的前提下才会生效，即环境变量[NPU_DEBUG](NPU_DEBUG.md)取值为"1"或"true"，开启方法如下：

    ```bash
    export NPU_DEBUG=1
    ```

- 该环境变量需要在import npu_device前设置。
- 该环境变量仅适用于TensorFlow  2.6.5网络在昇腾平台执行训练或在线推理的场景。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 推理系列产品

Atlas 训练系列产品
