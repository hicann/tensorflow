# Build Configurations

The following compilation configurations are required in the online inference script:

```python
import tensorflow as tf
import npu_bridge
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# Configuration 1: Schedule the inference job to the AI processor.
custom_op.parameter_map["use_off_line"].b = True

# Configuration 2: In the online inference scenario, you are advised to retain the default precision force_fp16 to achieve better performance.
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

# Configuration 3: Select the graph run mode. Set this parameter to 0 in the inference scenario or retain the default value 1 in the training scenario.
custom_op.parameter_map["graph_run_mode"].i = 0

# Configuration 4: Disable remapping and MemoryOptimizer.
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

```

The key configuration options in online inference are summarized as follows:

- Set **use_off_line** to **True**  to perform inference on the  AI processor.
- Retain the default  **precision_mode**  setting \(float16\) to achieve better performance.
- Set **graph_run_mode** to **0**.

The Ascend platform provides functions such as function debugging, performance optimization, and accuracy tuning. You can enable related functions by configuring the sessions. For details about the parameters, see  [Session Configuration](../../apiref/session_config/README.md).
