# set_device_sat_mode

## 功能说明

设置针对浮点计算的进程级溢出模式。

- 饱和模式：计算出现溢出时，计算结果会饱和为浮点数极值（+-MAX）。
- INF/NaN模式：遵循IEEE 754标准，根据定义输出INF/NaN的计算结果。

针对Ascend 950PR/Ascend 950DT，支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

针对Atlas 训练系列产品，默认值为“饱和模式”，且仅支持“饱和模式”。

## 函数原型

```python
def set_device_sat_mode(mode)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mode | 输入 | 设置的溢出模式。<br><br>  - 0：饱和模式。<br>  - 1：INF/NaN模式。<br><br>针对Ascend 950PR/Ascend 950DT，请保持默认值“1”。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，请保持默认值“1”。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，请保持默认值“1”。<br>针对Atlas 训练系列产品，默认值“0”，且仅支持配置为“0”。 |

## 返回值

无

## 约束说明

该接口需要在运行时配置，网络脚本执行前调用。

## 调用示例

以下示例仅针对Ascend 950PR/Ascend 950DT，Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品，其他型号处理器，开发者无需显式调用此接口。

```python
import tensorflow as tf
from npu_bridge.npu_init import *

......
# 网络执行时调用如下接口进行溢出模式的设置
npu_plugin.set_device_sat_mode(1)
sess.run(xxx)
```
