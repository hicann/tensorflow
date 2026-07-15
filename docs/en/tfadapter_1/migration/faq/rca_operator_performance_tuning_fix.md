# How Can I Achieve Expected Performance with Operators Including ResourceConditionalAccumulator?

## Symptom

The network \(such as OSMN\) fails to achieve satisfactory performance due to a large number of resource operators including ResourceConditionalAccumulator and ResourceAccumulatorTakeGradient.

## Possible Cause

Currently,  AI processor  uses the full offload mode by default. These operators show expensive scheduling and memory copy on  AI processor, resulting in unsatisfactory performance.

## Solution

Enable mixed computing to execute these operators on the host.

```python
from npu_bridge.npu_init import *

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b =  True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    sess.run(...)
```
