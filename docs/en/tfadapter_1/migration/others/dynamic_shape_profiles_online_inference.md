# Dynamic Shape Profiles (Online Inference)

## Background

During online inference, dynamic shape profiles can be specified in the user script to support dynamic-shape inputs.

This can be done by using the  **session**  parameter. The input can be a dataset, a placeholder, or a mixture of them. If a mixture of them is input, only one type of the inputs can be dynamic.

## Restrictions

- The input shapes set in the script must be in the same alphabetical order as the names of the data nodes. Assume that there are three inputs:  **label**,  **data**, and  **mask**.

    The input sequence of  **input_shape**  should be  **data**,  **label**, and  **mask**.

    ```text
    "data:1,1,40,-1;label:1,-1;mask:-1,-1"
    ```

- For dataset input, it is not allowed to specify a name in the  **get_next**  call. Otherwise, the system cannot identify whether the input is a dataset or a placeholder.
- A graph cannot contain more than one  **get_next**  node.
- This function is mutually exclusive with mixed computing.
- If both dataset and placeholder inputs are used, only one type of the inputs can be dynamic.

## Setting Dynamic Shape Profiles in sess.run Mode

```python
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,-1;label:1,-1;mask:-1,-1")
custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,20,1,1;40,40,2,2;80,60,4,4")
custom_op.parameter_map["dynamic_node_type"].i = 0
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Disable remapping.

with tf.Session(config=config) as sess:
  sess.run()
```

**input_shape**  indicates the input shapes. In this example, the network model has three inputs:  **data**  \(1, 1, 40, -1\),  **label**  \(1, -1\), and  **mask**  \(-1, -1\), where  **-1**  indicates the dimension featuring dynamic shape profiles, which need to be specified by  **dynamic_dims**.

**dynamic_dims**  indicates the dynamic shape profiles. Separate the shapes by semicolons \(;\). The dimension values match the  **-1**  placeholders in the  **input_shape**  argument with ordering preserved, and the number of  **-1**  placeholders equals the number of required dimensions in each profile. Based on the  **input_shape**  information, if  **dynamic_dims**  is set to  **"20,20,1,1;40,40,2,2;80,60,4,4"**, the meaning is as follows:

There are three semicolons \(;\), indicating that the input shape supports three profiles. The values in each profile correspond to  **-1**  in the input shape.

- Profile 0: data\(1,1,40,20\)+label\(1,20\)+mask\(1,1\)
- Profile 1: data\(1,1,40,40\)+label\(1,40\)+mask\(2,2\)
- Profile 2: data\(1,1,40,80\)+label\(1,60\)+mask\(4,4\)

    Assume that your graph has three inputs and only the first input has a static shape; the static shape must be specified in the  **input_shape**  field. For example:

    ```python
    custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,1;label:1,-1;mask:-1,-1")
    custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,1,1;40,2,2;60,4,4")
    ```

**dynamic_node_type**  sets the type of a dynamic input node. The value  **0**  indicates a dataset dynamic input while the value  **1**  indicates a placeholder dynamic input. Only one type of dynamic inputs is allowed, dataset or placeholder.
