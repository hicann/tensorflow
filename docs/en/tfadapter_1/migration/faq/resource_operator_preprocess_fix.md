# How Do I Fix the Data Preprocessing Error Caused by Resource Operators?

## Symptom

When the TensorFlow-based network is executed, the following error message is displayed:

```text
[2021-03-19 13:50:24.895266: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at lookup_table_op.cc:809 : Failed precondition: Table not initialized.
[2021-03-19 13:50:24.895283: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at lookup_table_op.cc:809 : Failed precondition: Table not initialized.
```

## Possible Cause

The resource operator HashTableV2 exists in the initialization graph, and the resource operator LookupTableFindV2 exists in the data preprocessing. The two operators must be used in pairs.

By default, the full offload mode is used on the  AI processor. That is, all compute operators \(including resource operators in initialization graphs\) are executed on the device, and data preprocessing is still executed on the host. In this case, the LookupTableFindV2 operator in data preprocessing and the HashTableV2 operator in the initialization graph are not executed on the same device. As a result, an error occurs during network execution.

## Solution

You need to modify the training script to enable mixed computing and execute the initialization graphs of resource operators on the host. An example of modifying the training script is as follows:

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

The  **mix_compile_mode**  parameter specifies whether to enable mixed computing. If this parameter is set to  **True**, resource operators that need to be used in pairs are executed online in the frontend framework.

Note: If the preprocessing script contains APIs of the  **Table**  class under  **tf.contrib.lookup**  that need to be used in pairs, you need to refer to this method to enable the mixed computing function and execute the corresponding operators in the initialization graphs on the host.
