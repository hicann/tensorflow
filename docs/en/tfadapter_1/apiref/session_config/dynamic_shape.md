# Dynamic Shape

> [!CAUTION]NOTICE
> In the scenario of dynamic dimension size profiles,  **input_shape**,  **dynamic_dims**, and  **dynamic_node_type**  must be used together.

## input_shape

Input shape. This parameter applies only to online inference scenarios.

Example:

```python
custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,-1;label:1,-1;mask:-1,-1")
```

In the preceding example, the network model has three inputs: data (1, 1, 40, –1), label (1, –1), and mask (–1, –1). Separate the name and shapes of each input with colons (:). –1 indicates a dynamic dimension, whose size profiles are configured by using dynamic_dims.

Notes:

- The names entered in input_shape must be in the same alphabetical order as the names of the actual data nodes. For example, for inputs data, label, and mask, the names entered in input_shape must be in the order of data, label, and mask.

- If a network has both dataset inputs and placeholder inputs, since dynamic inputs are described in only one of the modes, you only need to fill in the shapes of the dynamic inputs.
- For scalar inputs, set the shape to 0.
- The shape range specified by this parameter must be valid.

## dynamic_dims

Input dimension size profiles. This parameter applies only to online inference scenarios. Separate the dimension sizes by a semicolon (;). The dimension values match the –1 placeholders in the input_shape argument with ordering preserved, and the number of –1 placeholders equals the number of dimension sizes of each profile. Set at least two dynamic dimension size profiles.

The argument of dynamic_dims must match that of input_shape, as failure to do so may lead to an error and system's exit.

Example:

```python
custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,20,1,1;40,40,2,2;80,60,4,4")
```

Based on the input_shape information in the preceding example, the supported input shape profiles are as follows:

- Profile 0: data(1,1,40,20)+label(1,20)+mask(1,1)
- Profile 1: data(1,1,40,40), label(1,40), mask(2,2)
- Profile 2: data(1,1,40,80)+label(1,60)+mask(4,4)

Notes:

For the following products, the profile range is (1,100]. That is, at least two profiles must be set, and a maximum of 100 profiles are supported.

- Atlas A3 training product/Atlas A3 inference product
- Atlas A2 training product/Atlas A2 inference product
- Atlas inference product
- Atlas training product

For the Ascend 950PR/Ascend 950DT, the profile range is (1,256]. That is, at least two profiles must be set, and a maximum of 256 profiles are supported.

## dynamic_node_type

Type of a dynamic input node. This parameter applies only to online inference scenarios.

- 0: dataset input
- 1: placeholder input

Only one type of dynamic inputs is allowed, dataset or placeholder.

Example:

```python
custom_op.parameter_map["dynamic_node_type"].i = 0
```

## compile_hybrid_mode

Whether to enable the hybrid compilation and execution for dynamic dimension size profiles and dynamic shapes. This parameter applies only to online inference scenarios.

- 1: enabled. The graph is compiled into a dynamic dimension size profile graph and a dynamic shape graph. During execution, the input shape range is parsed. If the shape falls within the profile range, the profile graph is executed; otherwise, the dynamic shape graph is executed.
- 0 (default): disabled.

Note:

- This parameter must be used together with parameters for dynamic dimension size profiles (input_shape, dynamic_dims, and dynamic_node_type). Currently, only the scenario with dynamic_node_type set to 1 (placeholder inputs are dynamic) is supported.
- When compile_hybrid_mode is set to 1, if weight externalization for Const/Constant nodes is not enabled via external_weight, the system automatically enables this function to share weight files and reduce memory usage.

Example:

```python
custom_op.parameter_map["compile_hybrid_mode"].i = 1
```

## ac_parallel_enable

Whether to allow AI CPU operators and AI Core operators to run in parallel in a dynamic shape graph.

In a dynamic shape graph, when this parameter is enabled, the system automatically identifies AI CPU operators that can be concurrently executed with the AI Core operators in the graph. Operators of different engines are distributed to different flows to implement parallel execution among multiple engines, improving resource utilization and dynamic shape execution performance.

- 1: AI CPU operators and AI Core operators are allowed to run in parallel.
- 0 (default): AI CPU operators are not separately distributed.

Example:

```python
custom_op.parameter_map["ac_parallel_enable"].s = tf.compat.as_bytes("1")
```

## compile_dynamic_mode

Whether to generalize all input shapes in the graph.

- True: All input shapes are generalized to -1. Also, static shape graphs are generalized to dynamic ones.
- False (default): Input shapes are not generalized.

Example:

```python
custom_op.parameter_map["compile_dynamic_mode"].b = True
```

Note: This parameter cannot be used together with parameters for dynamic dimension size profiles (input_shape, dynamic_dims, and dynamic_node_type).

## all_tensor_not_empty

Whether to remove control nodes for empty tensor checks in the execution graph. In dynamic shape graph scenarios, control nodes are typically inserted to check whether a node is empty to prevent empty tensor nodes from being sent to the device. If you are certain that the graph does not contain empty tensors, you can enable this configuration to remove these control nodes and improve graph execution performance.

- True: Removes the control nodes used for empty tensor checks in the execution graph. Set it to True only when you are sure that the graph does not contain empty tensor nodes; otherwise, some operators may fail.
- False (default): Retains the control nodes used for empty tensor checks in the execution graph.

Example:

```python
custom_op.parameter_map["all_tensor_not_empty"].b = True
```
