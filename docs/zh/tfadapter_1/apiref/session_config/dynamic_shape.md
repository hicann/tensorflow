# 动态shape

> [!NOTE]说明
> 动态分档场景下，“input_shape”、“dynamic_dims”与“dynamic_node_type”三个参数需要配合使用。

## input_shape

此参数仅适用于在线推理场景，用户配置输入的shape信息。

配置示例：

```python
custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,-1;label:1,-1;mask:-1,-1")
```

上面示例中表示网络中有三个输入，输入的name分别为data，label，mask，各输入的shape分别为（1,1,40,-1），（1,-1），（-1,-1），name和shape之间以英文冒号分隔。其中-1表示该维度上为动态档位，需要通过dynamic_dims设置动态档位参数。

配置注意事项：

- input_shape中输入的name需要与实际data节点的name的字母顺序保持一致，例如有三个输入，顺序为：data、label、mask，则input_shape输入顺序应该为data、label、mask。

- 如果网络即有dataset输入也有placeholder输入，由于当前仅支持一种输入为动态的场景（例如dataset输入为动态），此时仅需填写dataset所有输入的shape信息。
- 如果输入中包含标量，则需要填写为0。
- 通过此参数设置的shape范围必须有效。

## dynamic_dims

此参数仅适用于在线推理场景，用于配置输入的对应维度的档位信息。档位中间使用英文分号分隔，每档中的dim值与input_shape参数中的-1标识的参数依次对应，input_shape参数中有几个-1，则每档必须设置几个维度。并且要求档位信息必须大于1组。

input_shape和dynamic_dims这两个参数的分档信息能够匹配，否则报错退出。

配置示例：

```python
custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,20,1,1;40,40,2,2;80,60,4,4")
```

结合上面举例的input_shape信息，表示支持输入的shape为：

- 第0档：data\(1,1,40,20\)，label\(1,20\)，mask\(1,1\)
- 第1档：data\(1,1,40,40\)，label\(1,40\)，mask\(2,2\)
- 第2档：data\(1,1,40,80\)，label\(1,60\)，mask\(4,4\)

需要注意：

针对如下产品，档位数取值范围为\(1,100\]，即必须设置至少2个档位，最多支持100档配置。

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 推理系列产品
- Atlas 训练系列产品

针对Ascend 950PR/Ascend 950DT，档位数取值范围为\(1, 256\]，即必须设置至少2个档位，最多支持256档配置。

## dynamic_node_type

此参数仅适用于在线推理场景，用于指定动态输入的节点类型。

- 0：dataset输入为动态输入。
- 1：placeholder输入为动态输入。

当前不支持dataset和placeholder输入同时为动态输入。

配置示例：

```python
custom_op.parameter_map["dynamic_node_type"].i = 0
```

## compile_hybrid_mode

此参数仅适用于在线推理场景，用于配置是否开启动态分档与动态shape混合编译执行功能。

- 1：开启，将图分别编译为动态分档图和动态shape图。执行时解析输入shape范围，若shape在分档范围内，则选择分档图执行；否则选择动态shape图执行。
- 0（默认值）：关闭此功能。

注意：

- 该参数必须与动态分档相关参数（“input_shape”、“dynamic_dims”和“dynamic_node_type”）同时使用，且当前仅支持“dynamic_node_type=1”的场景，即placeholder输入为动态输入的场景。
- “compile_hybrid_mode”设置为“1”时，若未通过“external_weight”开启Const/Constant节点的权重外置，系统将默认启用该功能，实现权重文件共享，从而降低内存占用。

配置示例：

```python
custom_op.parameter_map["compile_hybrid_mode"].i = 1
```

## ac_parallel_enable

动态shape图中，是否允许AI CPU算子和AI Core算子并行运行。

动态shape图中，开关开启时，系统自动识别图中可以和AI Core并发的AI CPU算子，不同引擎的算子下发到不同流上，实现多引擎间的并行，从而提升资源利用效率和动态shape执行性能。

- 1：允许AI CPU和AI Core算子间的并行运行。
- 0（默认值）：AI CPU算子不会单独分流。

 配置示例：

```python
custom_op.parameter_map["ac_parallel_enable"].s = tf.compat.as_bytes("1")
```

## compile_dynamic_mode

是否需要泛化图中所有的输入shape。

- True：将所有的输入shape泛化为-1，如果是静态shape图，则会泛化为动态shape图。
- False（默认值）：不泛化输入shape。

配置示例：

```python
custom_op.parameter_map["compile_dynamic_mode"].b = True
```

**注意**：此参数不能与动态分档相关参数同时使用，即不能与“input_shape”、“dynamic_dims”与“dynamic_node_type”三个参数同时使用。

## all_tensor_not_empty

动态shape计算图场景，为避免将空tensor节点下发到device，执行图通常会插入控制节点用于判断当前节点是否为空。如果用户确认计算图中不存在空tensor，可通过开启此配置移除这些控制节点，从而提升图执行性能。

- True：移除执行图中用于空tensor判断的控制节点。仅在确认计算图中不存在空tensor节点时开启，否则可能导致部分算子执行出错。
- False（默认值）：保留执行图中用于空tensor判断的控制节点。

配置示例：

```python
custom_op.parameter_map["all_tensor_not_empty"].b = True
```
