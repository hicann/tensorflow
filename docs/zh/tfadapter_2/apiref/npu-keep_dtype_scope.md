# npu.keep_dtype_scope

## 功能说明

指定哪些算子保持原有精度，如果原始网络模型中的算子精度在AI处理器上不支持，则系统内部自动采用算子支持的高精度来计算。

## 函数原型

```python
npu.keep_dtype_scope()
```

## 参数说明

无。

## 返回值

Python上下文管理器，在该上下文中的构图算子，会带有NPU识别的特殊属性。

## 调用示例

```python
import npu_device as npu
with npu.keep_dtype_scope():
    v = tf.add(1, 1)
```
