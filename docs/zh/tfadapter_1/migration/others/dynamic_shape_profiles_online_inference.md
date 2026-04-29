# 动态分档（在线推理）

## 背景介绍

**在线推理场景下**，当前系统支持在用户脚本中指定配置动态档位信息，从而支持动态输入的场景。

开发者可以使用session参数设置分档信息，输入可以为dataset方式、placeholder方式，或者两种混合方式。对于混合输入，当前仅支持其中一种为动态变化的场景。

## 使用约束

- 用户在脚本中设置的input shape的输入顺序要和实际data节点的name字母序保持一致，例如有三个输入：label、data、mask，则

    input_shape输入顺序应该为data、label、mask：

    ```text
    "data:1,1,40,-1;label:1,-1;mask:-1,-1"
    ```

- 对于dataset输入时，get_next接口不允许指定name，否则系统无法识别该输入为dataset输入还是placeholder输入。
- 不支持同一张图中有多个get_next节点。
- 该功能不能和混合计算一起使用。
- 对于dataset和placeholder混合输入，当前仅支持其中一种为动态变化的场景。

## sess.run模式下设置动态分档

```python
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,-1;label:1,-1;mask:-1,-1")
custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,20,1,1;40,40,2,2;80,60,4,4")
custom_op.parameter_map["dynamic_node_type"].i = 0
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关

with tf.Session(config=config) as sess:
  sess.run()
```

input_shape表示网络的输入shape信息，以上配置表示网络中有三个输入，输入的name分别为data，label，mask，各输入的shape分别为（1,1,40,-1），（1,-1），（-1,-1），其中-1表示该维度上为动态档位，需要通过dynamic_dims设置动态档位参数。

dynamic_dims表示输入的对应维度的档位信息。档位中间使用英文分号分隔，每档中的dim值与input_shape参数中的-1标识的参数依次对应，input_shape参数中有几个-1，则每档必须设置几个维度。结合input_shape信息，dynamic_dims配置为"20,20,1,1;40,40,2,2;80,60,4,4"的含义如下：

有三个“;”，表示输入shape支持三个档位，每个档位中的值对应输入shape中的“-1”的取值：

- 第0档：data\(1,1,40,20\)，label\(1,20\)，mask\(1,1\)
- 第1档：data\(1,1,40,40\)，label\(1,40\)，mask\(2,2\)
- 第2档：data\(1,1,40,80\)，label\(1,60\)，mask\(4,4\)

    对于多输入场景，例如有三个输入时，如果只有第二个第三个输入是动态档位，第一个输入为固定输入时，仍需要将固定输入shape填入input_shape字段内，例如：

    ```python
    custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,1;label:1,-1;mask:-1,-1")
    custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,1,1;40,2,2;60,4,4")
    ```

dynamic_node_type用于指定动态输入的节点类型。0：dataset输入为动态输入；1：placeholder输入为动态输入。当前不支持dataset和placeholder输入同时为动态输入。
