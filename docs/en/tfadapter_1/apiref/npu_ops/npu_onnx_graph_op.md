# npu_onnx_graph_op

## Description

Loads an ONNX model as an operator and executes it on the  AI processor  through the TensorFlow framework.

![](../figures/npu_onnx_graph_op.png)

## Prototype

```python
def npu_onnx_graph_op(inputs, tout, model_path, name=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| inputs | Input | Input tensor of the ONNX model. |
| tout | Input | Output data type of the ONNX model, for example, tf.float32. |
| model_path | Input | A string for the path (including the file name) of the ONNX model, for example, /test/test.onnx. |
| name | Input | A string for the name of the operator on the graph. |

## Returns

Operator output list.

## Example

```python
from npu_bridge.estimator.npu_ops import npu_onnx_graph_op

input = tf.placeholder(dtype=tf.float32, shape=(1, 1, 5, 5), name="conv_input")
output = npu_onnx_graph_op([input], [tf.float32], model_path="conv2d.onnx", name="conv2d")
```
