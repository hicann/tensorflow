# 简介

## 功能说明

TF Adapter提供了系列session配置用于进行功能调试、性能提升、精度提升等，开发者在AI处理器上进行模型训练或在线推理时，可以使用这些session配置。

您可以在TensorFlow Adapter软件安装路径下的：python/site-packages/npu_bridge/estimator/npu/npu_estimator.py文件中查看相关配置定义，如果相关参数本章节未列出，表示该参数预留或适用于其他AI处理器版本，用户无需关注。

## 调用示例

session配置的通用使用方式如下所示：

```python
import tensorflow as tf
from npu_bridge.npu_init import *
...
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
...
with tf.Session(config=config) as sess:
    sess.run(cost)
```
