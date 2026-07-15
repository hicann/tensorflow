# subgraph_multi_dims_scope

## Description

Specifies the scope of the operator for which subgraph-wide dynamic shape profiles are to be applied in the online inference scenario.

**Note: This API is a trial API and should not be directly used by developers.**

## Prototype

```python
def subgraph_multi_dims_scope(index)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| index | Input | Scope index. When multiple scopes are to be specified by using this API, ensure that each index is unique. |

## Returns

None

## Restrictions

If the graph-wide dynamic dimension size is configured, this setting is invalid.

Scope embedding is not supported.

## Example

```python
ori_image = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3), name="ori_image")
resized_img, h_scale, w_scale = npu_cpu_ops.ocr_detection_pre_handle(img=ori_image)

with npu_scope.subgraph_multi_dims_scope(0):
    image_expand = tf.expand_dims(resized_img, axis=0)
    util.set_op_input_tensor_multi_dims(image_expand, "0:-1,-1,3", "480,480;960,960;1920,1920")
    image_tensor_fp32 = tf.cast(image_expand, dtype=tf.float32)
    image_tensor_nchw = tf.transpose(image_tensor_fp32, [0, 3, 1, 2])
    score, kernel = npu_onnx_graph_op([image_tensor_nchw], [tf.float32, tf.uint8], model_path="text_detection.onnx", name="detection")

with tf.Session(config=config) as sess:
    sess.run()
```

The argument 0 of  **subgraph_multi_dims_scope**  indicates the scope index. If there are multiple scopes in the graph, the value must be unique.

**set_op_input_tensor_multi_dims**  has three inputs:

- The first input is any output tensor of the input node within the scope.
- The second input configures all input shapes of the input node corresponding to the tensor. Inputs are separated by semicolons \(;\), input indices and shapes are separated by colons \(:\), and dims in a shape are separated by commas \(,\).
- The third input is the profiles information. Profiles are separated by semicolons \(;\), and dims in each profile are separated by commas \(,\). The number of dims is the same as the number of  **-1**  in  **input_shape**.

The preceding configuration indicates that the input node of the scope is expand_dims, which has one dynamic input having shape \(-1, -1, 3\). Three profiles are available:

- Profile 0: \(480,480,3\)
- Profile 1: \(960,960,3\)
- Profile 2: \(1920,1920,3\)
