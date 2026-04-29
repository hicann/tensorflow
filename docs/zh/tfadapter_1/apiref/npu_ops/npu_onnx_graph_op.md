# npu_onnx_graph_op

## 功能说明

以算子形式加载ONNX模型，将指定路径中onnx模型通过TensorFlow的框架执行在AI处理器。

![](../figures/npu_onnx_graph_op.png)

## 函数原型

```python
def npu_onnx_graph_op(inputs, tout, model_path, name=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| inputs | 输入 | ONNX模型的输入，类型为tensor。 |
| tout | 输入 | ONNX模型的输出类型，如tf.float32。 |
| model_path | 输入 | ONNX模型所在路径及文件名，例如/test/test.onnx，类型string。 |
| name | 输入 | 指定该算子在图上的名称，类型string。 |

## 返回值

算子输出列表。

## 调用示例

```python
from npu_bridge.estimator.npu_ops import npu_onnx_graph_op

input = tf.placeholder(dtype=tf.float32, shape=(1, 1, 5, 5), name="conv_input")
output = npu_onnx_graph_op([input], [tf.float32], model_path="conv2d.onnx", name="conv2d")
```
