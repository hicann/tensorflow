# Initializing Collective Communication

Before using HCCL APIs, initialize collective communication first. The initialization is hidden in the  **initialize_system**  API. If you use an HCCL API such as  **get_local_rank_id**,  **get_rank_size**, or  **get_rank_id**  before  **sess.run\(\)**  or  **estimator.train\(\)**, you need to start another session and execute  **initialize_system**  to initialize collective communication. After the training is complete, execute  **shutdown_system**  and close the session.

Note that if these HCCL APIs are called after the  **sess.run\(\)**  or  **estimator.train\(\)**  call, initialization needs to be performed again because the session for collective communication initialization will be closed automatically.

The following is a sample code:

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

# Example of graph execution logic
a = tf.placeholder(tf.int32, (None,None))
b = tf.placeholder(tf.int32, (None,None))
add = tf.add(a, b)

with tf.Session(config=config) as sess:
    # Initialize collective communication.
    sess.run(npu_init)

    # <!---- Call collective communication APIs. Fill in the code as required. ---->

    # Example of training
    result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],b: [[1],[-21]]})
    # Close the session.
    sess.run(npu_shutdown)
```
