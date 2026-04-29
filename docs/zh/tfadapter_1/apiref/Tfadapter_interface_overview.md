# TF Adapter接口简介

## 简介

用户可以基于深度学习框架TensorFlow 1.15进行训练或在线推理脚本的开发，TF Adapter提供了适配TensorFlow 1.15框架的用户接口。

**图 1**  TF Adapter接口  
![TF-Adapter接口](figures/TF-Adapter_interface.png)

接口路径：$\{TFPLUGIN_INSTALL_PATH\}/python/site-packages/npu_bridge。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 训练系列产品
- Atlas 推理系列产品（仅支持在线推理特性）

> [!NOTE]说明
> 针对Atlas 推理系列产品，仅支持在线推理特性。
