# 数据预处理中存在资源类算子导致训练异常

## 问题现象

TensorFlow网络执行时，报如下错误：

```text
[2021-03-19 13:50:24.895266: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at lookup_table_op.cc:809 : Failed precondition: Table not initialized.
[2021-03-19 13:50:24.895283: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at lookup_table_op.cc:809 : Failed precondition: Table not initialized.
```

## 原因分析

初始化图中存在资源类算子HashTableV2 ，数据预处理中存在资源类算子LookupTableFindV2，两个算子需要配对使用。

AI处理器默认采用计算全下沉模式，即所有的计算类算子（包括初始化图中的资源类算子）全部在Device侧执行，数据预处理仍在Host执行。这样数据预处理中的LookupTableFindV2算子与初始化图中的HashTableV2算子未在同一设备上执行，导致网络运行出错。

## 解决方案

需要修改训练脚本，开启混合计算能力，将资源类算子的初始化图也留在Host侧执行，训练脚本修改方法的示例如下：

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

其中配置参数“**mix_compile_mode**”是混合计算开启开关，当此开关配置为“True”后，会将需要成对使用的资源类算子留在前端框架在线执行。

**补充说明：当用户的预处理脚本中存在需要成对使用的tf.contrib.lookup下Table类的API时，需要参考此方法使能混合计算功能，将初始化图中的对应算子留在Host侧执行。**
