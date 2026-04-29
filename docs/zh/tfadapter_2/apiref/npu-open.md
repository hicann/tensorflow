# npu.open

## 功能说明

用于注册NPU设备，当前必须连续调用as_default设置NPU为默认设备。

## 函数原型

```python
npu.open(device_id=None).as_default()
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| device_id | 输入 | 需要初始化的NPU设备ID，如果不传入，默认读取环境变量ASCEND_DEVICE_ID的值，如果ASCEND_DEVICE_ID环境变量未设置，则取值为0。<br>ASCEND_DEVICE_ID用于指定当前进程所用的AI处理器逻辑ID，取值范围[0,N-1]，设置示例：<br>`export ASCEND_DEVICE_ID=0` |

## 返回值

返回npu_device的实例。

## 调用示例

```python
import npu_device as npu
npu.open().as_default()
```
