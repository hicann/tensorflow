# Overview

## Function Description

TF Adapter provides a series of session configurations for function debugging, performance improvement, and precision improvement. Developers can use these session configurations when performing model training or online inference on the  AI processor.

You can view related configuration definitions in the  **python/site-packages/npu_bridge/estimator/npu/npu_estimator.py**  file in the TensorFlow Adapter installation directory. The parameters that are not listed in this section are reserved or applicable to other  AI processor  versions.

## Examples

The usage of session configurations is as follows:

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
