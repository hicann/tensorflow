# without_npu_compile_scope

## 功能说明

混合计算场景下，配置在Host侧编译的算子。

## 函数原型

```python
def without_npu_compile_scope()
```

## 参数说明

无

## 约束说明

由于variable类算子必须要在device侧执行，因此该接口不支持配置variable类算子。

## 返回值

无

## 调用示例

请参见[混合计算](../../migration/others/mixed_computing.md)。
