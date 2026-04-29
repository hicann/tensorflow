# 网络中存在ResourceConditionalAccumulator等算子导致训练性能不达标

## 问题现象

OSMN等网络中存在大量的ResourceConditionalAccumulator、ResourceAccumulatorTakeGradient资源类算子，导致训练性能不达标。

## 原因分析

当前AI处理器默认采用计算全下沉模式，这些算子在AI处理器上执行时调度开销和内存拷贝开销大，导致训练性能不达标。

## 解决方案

需要开启混合计算能力，将此类算子留在Host侧执行。

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
