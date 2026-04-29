# 混合计算

## mix_compile_mode

是否开启混合计算模式。

- True：开启。
- False：关闭，默认关闭。

计算全下沉模式即所有的计算类算子全部在Device侧执行，混合计算模式作为计算全下沉模式的补充，将部分不可离线编译下沉执行的算子留在前端框架中在线执行，提升AI处理器支持TensorFlow的适配灵活性。

配置示例：

```python
custom_op.parameter_map["mix_compile_mode"].b =  True
```

## in_out_pair_flag

此参数仅适用于在线推理场景，用于混合计算场景下，配置是否将in_out_pair中指定的算子下沉到AI处理器执行，取值：

- True：下沉，默认为True。
- False：不下沉。

配置示例：

```python
custom_op.parameter_map['in_out_pair_flag'].b = False
```

## in_out_pair

此参数仅适用于在线推理场景，用于在混合计算场景下配置下沉/不下沉部分的首尾算子名。

需要注意，此参数仅支持配置一个\[in_nodes,out_nodes\]范围段内的算子，不支持配置多个\[in_nodes,out_nodes\]范围段。

配置示例：

```python
# 开启混合计算
custom_op.parameter_map["mix_compile_mode"].b = True
# 如下配置，将in_nodes, out_nodes范围内的算子全部下沉到NPU执行，其余算子留在前端框架执行。
in_nodes.append('import/conv2d_1/convolution')
out_nodes.append('import/conv2d_59/BiasAdd')
out_nodes.append('import/conv2d_67/BiasAdd')
out_nodes.append('import/conv2d_75/BiasAdd')
all_graph_iop.append([in_nodes, out_nodes])
custom_op.parameter_map['in_out_pair'].s = tf.compat.as_bytes(str(all_graph_iop))
# 或者通过如下配置，将in_nodes, out_nodes范围内的算子不下沉，全部留在前端框架执行，其余算子下沉到NPU执行。
in_nodes.append('import/conv2d_1/convolution')
out_nodes.append('import/conv2d_59/BiasAdd')
out_nodes.append('import/conv2d_67/BiasAdd')
out_nodes.append('import/conv2d_75/BiasAdd')
all_graph_iop.append([in_nodes, out_nodes])
custom_op.parameter_map['in_out_pair_flag'].b = False
custom_op.parameter_map['in_out_pair'].s = tf.compat.as_bytes(str(all_graph_iop))
```
