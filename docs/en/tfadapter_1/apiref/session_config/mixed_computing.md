# Mixed Computing

## mix_compile_mode

Mixed computing

- True: enabled.
- False (default): disabled.

In full offload mode, all compute operators are offloaded to the device. As a supplement to the full offload mode, mixed computing allows certain operators to be executed online within the frontend framework, improving the AI processor's adaptability to TensorFlow.

Example:

```python
custom_op.parameter_map["mix_compile_mode"].b =  True
```

## in_out_pair_flag

Whether to offload operators specified by in_out_pair to the AI processor in mixed computing scenarios. This parameter applies only to online inference scenarios. The values are as follows:

- True (default)
- False

Example:

```python
custom_op.parameter_map['in_out_pair_flag'].b = False
```

## in_out_pair

Names of the input-layer and output-layer operators offloaded (or not) in mixed computing scenarios. This parameter applies only to online inference scenarios.

Note that this parameter supports only one operator configured within the range of [in_nodes, out_nodes].

Example:

```python
# Enable mixed computing.
custom_op.parameter_map["mix_compile_mode"].b = True
# Offload operators within the [in_nodes, out_nodes] range to the NPU for execution, and execute other operators in the frontend framework.
in_nodes.append('import/conv2d_1/convolution')
out_nodes.append('import/conv2d_59/BiasAdd')
out_nodes.append('import/conv2d_67/BiasAdd')
out_nodes.append('import/conv2d_75/BiasAdd')
all_graph_iop.append([in_nodes, out_nodes])
custom_op.parameter_map['in_out_pair'].s = tf.compat.as_bytes(str(all_graph_iop))
# Alternatively, retain operators within the [in_nodes, out_nodes] range for execution in the frontend framework, and offload other operators to the NPU for execution.
in_nodes.append('import/conv2d_1/convolution')
out_nodes.append('import/conv2d_59/BiasAdd')
out_nodes.append('import/conv2d_67/BiasAdd')
out_nodes.append('import/conv2d_75/BiasAdd')
all_graph_iop.append([in_nodes, out_nodes])
custom_op.parameter_map['in_out_pair_flag'].b = False
custom_op.parameter_map['in_out_pair'].s = tf.compat.as_bytes(str(all_graph_iop))
```
