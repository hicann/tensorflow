# 简介

## 功能说明

返回NPU设备初始化全局单例配置对象，通过修改该全局单例对象的属性值，可以控制NPU设备初始化时的选项。该接口必须在调用npu.open接口初始化NPU设备前完成设置。

本节描述NPU提供的全局单例相关配置。

## 函数原型

npu.global_options\(\)

## 调用示例

如果您需要修改默认配置项，应当在初始化NPU设备前设置全局配置项，例如如果需要将精度模式由allow_fp32_to_fp16修改为allow_mix_precision，调用示例如下所示：

```python
import npu_device as npu
npu.global_options().precision_mode = 'allow_mix_precision'
npu.open().as_default()
```
