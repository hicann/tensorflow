# 脚本执行

## 准备数据集

准备数据集并上传到运行环境的目录下，例如：/home/data/resnet50/imagenet。

## 准备rank table文件

rank table文件样例和文件说明请参考《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》的“相关参考 \> 集群信息配置”章节。

## 配置环境变量

请参考[执行单Device训练](../model_training/single_device_training.md)配置环境变量。

## 执行命令

```bash
python3 /home/official/r1/resnet/imagenet_main.py --batch_size=32 --hooks=ExamplesPerSecondHook --data_dir=/home/data/resnet50/imagenet
```
