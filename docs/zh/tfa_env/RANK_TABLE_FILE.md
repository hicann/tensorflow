# RANK_TABLE_FILE

## 功能描述

TensorFlow分布式训练或推理场景下，通过此环境变量指定参与集合通信的AI处理器的rank table资源配置文件，包含rank table文件路径和文件名。

关于rank table配置文件的说明可参见《[HCCL集合通信库](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“相关参考 \> 集群信息配置”章节。

## 配置示例

```bash
export RANK_TABLE_FILE=/home/test/ranktable.json
```

## 使用约束

无

## 支持的型号

Ascend 950PR&950DT系列产品

Atlas A3系列产品

Atlas A2系列产品

Atlas训练系列产品

Atlas 300I Duo推理卡
