# ASCEND_DEVICE_ID

## 功能描述

指定当前进程所用的AI处理器的逻辑ID。

取值范围\[0,N-1\]，默认为0。其中N为当前物理机/虚拟机/容器内的设备总数。

**该环境变量使用场景：**

TensorFlow框架网络在昇腾平台执行训练或在线推理的场景。

## 配置示例

```bash
export ASCEND_DEVICE_ID=0
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 推理系列产品

Atlas 训练系列产品
