# npu.set_npu_loop_size

## 功能说明

用于设置NPU循环下沉执行时的循环次数。

## 函数原型

```python
npu.set_npu_loop_size(loop_size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| loop_size | 输入 | NPU循环下沉执行时的循环次数，必须为大于0的整数。 |

## 返回值

无。

## 调用示例

```python
import npu_device as npu
npu.set_npu_loop_size(100) # 设置循环下沉次数为100
```
