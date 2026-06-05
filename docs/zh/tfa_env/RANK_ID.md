# RANK_ID

## 功能描述

TensorFlow分布式训练或推理场景下，通过此环境变量指定当前进程在集合通信进程组中对应的rank标识。

## 配置示例

```bash
export RANK_ID=0
```

## 使用约束

该环境变量的取值需要与rank table文件中的rank_id字段保持一致，关于rank table配置文件的说明可参见《[HCCL集合通信库](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“相关参考 \> 集群信息配置”章节。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 训练系列产品

Atlas 300I Duo 推理卡
