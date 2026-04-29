# keep_tensors_dtypes

## 功能说明

指定哪些算子保持原有精度。

## 函数原型

```python
def keep_tensors_dtypes(graph, input_tensors)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graph | 输入 | 从pb模型导入的图。 |
| input_tensors | 输入 | 需要保持精度的算子名称。 |

## 返回值

无

## 使用约束

- 该接口仅适用于在线推理场景。
- 算子精度模式为保持原图精度（即precision_mode指定为must_keep_origin_dtype）时，该接口不生效。

## 调用示例

```python
from npu_bridge.estimator.npu import util
g=tf.Graph()
util.keep_tensors_dtypes(g,("random_uniform_1/sub:0",))
```
