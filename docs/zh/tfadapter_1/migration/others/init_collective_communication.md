# 集合通信初始化

使用集合通信接口前首先需要进行集合通信初始化，当前集合通信的初始化隐藏在initialize_system接口中，如果在sess.run或者estimator.train之前调用get_local_rank_id/get_rank_size/get_rank_id等集合通信接口，需要另外启动session执行initialize_system，进行集合通信初始化，然后在训练结束后执行shutdown_system，同时关闭session。

**需要注意**：如果在sess.run或者estimator.train之后又调用了集合通信接口，由于sess.run或者estimator.train后系统会自动关闭集合通信初始化session，因此需要再次进行集合通信初始化。

代码片段示例：

```python
import tensorflow as tf
from npu_bridge.npu_init import *

npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

# 图执行逻辑，此处仅为一个简单示例
a = tf.placeholder(tf.int32, (None,None))
b = tf.placeholder(tf.int32, (None,None))
add = tf.add(a, b)

with tf.Session(config=config) as sess:
    # 进行集合通信初始化
    sess.run(npu_init)

    # <!---- 调用集合通信接口，请根据实际需要填充代码 ---->

    # 执行训练，此处仅为示例
    result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],b: [[1],[-21]]})
    # 关闭session
    sess.run(npu_shutdown)
```
