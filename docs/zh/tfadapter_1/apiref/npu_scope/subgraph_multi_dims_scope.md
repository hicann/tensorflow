# subgraph_multi_dims_scope

## 功能说明

在线推理场景下指定需要进行子图动态分档的算子scope。

**注意：此接口是试验接口，不建议开发者直接使用。**

## 函数原型

```python
def subgraph_multi_dims_scope(index)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| index | 输入 | scope索引，当一张图中多次使用该接口指定分档scope时，需要保证index唯一。 |

## 返回值

无

## 约束说明

如果配置了整图动态分档，该功能失效。

不支持scope嵌套。

## 调用示例

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

subgraph_multi_dims_scope的入参0表示scope索引，如果图中出现多个scope，该值需要保证唯一。

set_op_input_tensor_multi_dims有三个输入：

- 第一个输入为scope范围内输入节点的任意输出tensor；
- 第二个输入配置该tensor对应的节点的所有输入shape，每个输入间用";"分隔，input索引与shape间用":"分隔，shape的多个dim间用","分隔；
- 第三个输入为档位信息，每个档位间以";"分隔，档位内dim间用","分隔，dim数与input_shape中"-1"的数量一致。

以上配置表示分档范围的输入节点是expand_dims，该算子有1个输入，为动态shape：\(-1,-1,3\)，可分为3个档位：

- 第0档：\(480,480,3\)
- 第1档：\(960,960,3\)
- 第2档：\(1920,1920,3\)
