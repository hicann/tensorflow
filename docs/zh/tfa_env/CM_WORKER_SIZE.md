# CM_WORKER_SIZE

## 功能描述

TensorFlow分布式训练或推理场景下，用户可以选择不使用rank table文件，通过组合使用环境变量[CM_CHIEF_IP](CM_CHIEF_IP.md)、[CM_CHIEF_PORT](CM_CHIEF_PORT.md)、[CM_CHIEF_DEVICE](CM_CHIEF_DEVICE.md)、[CM_WORKER_SIZE](CM_WORKER_SIZE.md)、[CM_WORKER_IP](CM_WORKER_IP.md)的方式自动生成资源信息，完成集合通信组件初始化。

本环境变量“CM_WORKER_SIZE”用于配置本次业务通信域内Device的总数量。

该环境变量需要配置为整数，取值范围“0~32768”。

## 配置示例

```bash
export CM_WORKER_SIZE=8
```

## 使用约束

此环境变量不能与[RANK_TABLE_FILE](RANK_TABLE_FILE.md)、[RANK_ID](RANK_ID.md)、[RANK_SIZE](RANK_SIZE.md)混合使用。

## 支持的型号

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 训练系列产品
