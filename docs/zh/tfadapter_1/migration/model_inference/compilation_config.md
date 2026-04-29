# 编译配置

在线推理脚本中需要进行必要的编译配置：

```python
import tensorflow as tf
import npu_bridge
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# 配置1：选择在AI处理器上执行推理
custom_op.parameter_map["use_off_line"].b = True

# 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

# 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
custom_op.parameter_map["graph_run_mode"].i = 0

# 配置4：关闭remapping和MemoryOptimizer
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

```

在线推理几个关键配置项为：

- use_off_line配置为True，表示在AI处理器上执行推理。
- precision_mode建议保持默认，使用float16精度推理，以获得较优的性能。
- graph_run_mode配置为0。

Ascend平台提供了功能调试、性能调优/精度调试等功能，开发者可通过对应的session配置的方式使能相关功能，详细的参数说明可参见[session配置](../../apiref/session_config/README.md)。
