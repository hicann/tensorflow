# Parameters That Will Be Deprecated in Later Versions

The following parameters will be deprecated in later versions. You are advised not to use them anymore.

## enable_data_pre_proc

Performance tuning. Enable for the GetNext operator offload to the NPU. The GetNext operator offload is a prerequisite for iteration offload.

- True (default): enabled. The prerequisite for GetNext operator offload is that the TensorFlow Dataset mode is used to read data.
- False: disabled.

Example:

```python
config = NPURunConfig(enable_data_pre_proc=True)
```

## variable_format_optimize

Performance tuning. Variable format optimization enable.

- True: enabled.
- False: disabled.

To improve training efficiency, the format of the variables is converted to a format more compatible with the AI processor during variable initialization performed by the network. Enable or disable this function as needed.

This parameter is left empty by default, indicating that the configuration is disabled.

Example:

```python
config = NPURunConfig(variable_format_optimize=True)
```

## op_debug_level

Operator debug enable. The values are as follows:

- 0: disables operator debug.
- 1: Enables operator debug. TBE instruction mapping files are generated in the kernel_meta directory under the training script execution path, including operator CCE files (.cce), Python-CCE mapping files (_loc.json), .o files, and .json files. These files are used for AI Core error analysis with related tools.

  Note: For the Ascend 950PR/Ascend 950DT, no TBE instruction mapping files are generated.

- 2: Enables operator debug. TBE instruction mapping files are generated in the kernel_meta directory under the training script execution path, including operator CCE files (.cce), Python-CCE mapping files (_loc.json), .o files, and .json files. The compilation optimization of the CCE compiler is disabled and the CCE compiler debugging function is enabled (by setting the compiler option to -O0-g). These files are used for AI Core error analysis with related tools.

  Note: For the Ascend 950PR/Ascend 950DT, no TBE instruction mapping files are generated.

- 3: disables operator debug. The operator .o and .json files are retained in the kernel_meta folder in the training script execution directory.
- 4: disables operator debug. The operator binary (.o) and operator description file (.json) are retained, and a TBE instruction mapping file (.cce) and a UB fusion description file ({$kernel_name}_compute.json) are generated in the kernel_meta folder under the training script execution directory.

  For the Ascend 950PR/Ascend 950DT, neither TBE instruction mapping files nor UB fusion compute description files are generated.

  NOTICE:

  - If this option is set to 0 and op_debug_config is configured, the operator compilation directory kernel_meta is still generated in the current execution path during training. The content generated in the directory is subject to op_debug_config.
  - You are advised to set this option to 0 or 3 for training. To locate AI Core errors, set this parameter to 1 or 2, which might compromise the network performance.
  - If this option is set to 2 (the CCE compiler is enabled), it cannot be used together with the oom option in op_debug_config. Otherwise, an AI Core error is reported. The following is an example of the error message:
  
  ```text
  ...there is an aivec error exception, core id is 49, error code = 0x4 ...
  ```

  - If this parameter is set to 2 (the CCE compiler is enabled), the size of the operator kernel file (*.o file) increases. In dynamic shape scenarios, all possible scenarios are traversed during operator build, which may cause operator build failures due to large operator kernel files. In this case, 2 is not recommended.
  
    If the build failure is caused by the large operator kernel file, the following log is displayed:

    ```text
    message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
    ```

- If the value of this parameter is not 0, you can use the debug_dir parameter to specify the path for storing debugging-related process files.
- If this parameter is set to 0 and NPU_COLLECT_PATH is set, the operator compilation directory kernel_meta is generated in the current path after the command is executed. If ASCEND_WORK_PATH is set, kernel_meta is generated in the path specified by the environment variable. For details about environment variables, see [Environment Variables](https://www.hiascend.com/document/detail/en/canncommercial/900/maintenref/envvar/envref_07_0001.html).
- When the debug function is enabled, if the model contains the following merged compute and communication (MC2) operators, the *.o,*.json, and *.cce files of the operators are not generated in the operator build folder kernel_meta.MatMulAllReduce

    ```text
    MatMulAllReduce
    MatMulAllReduceAddRmsNorm
    AllGatherMatMul
    MatMulReduceScatter
    AlltoAllAllGatherBatchMatMul
    BatchMatMulReduceScatterAlltoAll
    ```

This parameter is left empty by default, indicating that the configuration is disabled.

Example:

```python
config = NPURunConfig(op_debug_level=1)
```

## op_select_implmode

Operator implementation mode. Certain operators built in the NPU can be implemented in either high-precision or high-performance mode at model build time. Arguments:

- high_precision: high-precision implementation mode. In high-precision mode, Taylor's theorem or Newton's method is used to improve operator precision with float16 input.
- high_performance (default): high-performance implementation mode. In high-performance mode, the optimal performance is implemented without affecting the network precision (float16).

This parameter is left empty by default, indicating that the configuration is disabled.

Example:

```python
config = NPURunConfig(op_select_implmode="high_precision")
```

## optypelist_for_implmode

List of operator types (separated by commas) that use the mode specified by the op_select_implmode parameter. Currently, Pooling, SoftmaxV2, LRN, and ROIAlign operators are supported.

Use this parameter in conjunction with op_select_implmode, for example:

```python
config = NPURunConfig(
    op_select_implmode="high_precision",
    optypelist_for_implmode="Pooling,SoftmaxV2")
```

This parameter is left empty by default, indicating that the configuration is disabled.

## dynamic_input

Whether it is a dynamic input.

- True: dynamic input.
- False (default): static input.

Example:

```python
config = NPURunConfig(dynamic_input=True)
```

## dynamic_graph_execute_mode

Execution mode of a dynamic input. That is, this option takes effect when dynamic_input is set to True. Possible values are:

dynamic_execute: dynamic graph compilation. In this mode, the shape range configured in dynamic_inputs_shape_range is used for compilation.

Example:

```python
config = NPURunConfig(dynamic_graph_execute_mode="dynamic_execute")
```

## dynamic_inputs_shape_range

Shape range of each dynamic input. If a graph has two dataset inputs and one placeholder input, a configuration example is as follows:

```python
config = NPURunConfig(dynamic_inputs_shape_range="getnext:[128 ,3~5, 2~128, -1],[64 ,3~5, 2~128, -1];data:[128 ,3~5, 2~128, -1]")
```

Precautions:

- getnext indicates the dataset inputs and data indicates the placeholder inputs.
- The size of a static dimension is specified by a determinant value. The size range of a dynamic dimension is specified by using a tilde (~). A dynamic dimension without size range specified is denoted by –1.
- Assume that your graph has three dataset inputs but the first dataset input has a static shape; the static shape must be specified as shown below.

  ```python
  config = NPURunConfig(dynamic_inputs_shape_range="getnext:[3,3,4,10],[-1,3,2~1000,-1],[-1,-1,-1,-1]")
  ```

- For scalar inputs, you also need to fill in the shape range by using square brackets ([]). No space is allowed before [].
- If there are multiple getnext inputs or data inputs on the network, the input ordering must be preserved. For example:

  - If there are multiple dataset inputs on the network:def func(x):

    ```python
    def func(x):
        x = x + 1
        y = x + 2
        return x,y
    dataset = tf.data.Dataset.range(min_size, max_size)
    dataset = dataset.map(func)
    ```

    Assume that the first input of the network is x (with shape range [3~5]) and the second input is y (with shape range [3~6]). When configuring the dynamic ranges in dynamic_inputs_shape_range, the ordering must be preserved.

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range ="getnext:[3~5],[3~6]")
    ```

  - If there are multiple placeholder inputs on the network:
  
    If the placeholder names are not specified, for example:

    ```python
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    ```

   Set the dynamic ranges of the placeholder inputs in dynamic_inputs_shape_range in the same order as that defined in the script. That is, the first input x (with shape range [3~5]) goes first and the second input y (with shape range [3~6]) follows.

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range= "data:[3~5],[3~6]")
    ```

   If the placeholder names are specified, for example:

    ```python
    x = tf.placeholder(tf.int32, name='b')
    y = tf.placeholder(tf.int32, name='a')
    ```

   The inputs are in the alphabetical order of the name fields,

   that is, when setting dynamic_inputs_shape_range, the first input y (with shape range [3~6]) goes first and the second input x (with shape range [3~5]) follows.

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range = "data:[3~6],[3~5]")
    ```

    NOTICE:
    - For subgraphs with different input shapes, [set_graph_exec_config](../../npu_util/set_graph_exec_config.md) is recommended for supporting dynamic inputs. dynamic_inputs_shape_range applies only to a single graph, which may cause execution errors.
    - If the placeholder names are not specified in the network script, the placeholders are named in the following format:

      xxx_0, xxx_1, xxx_2, ......

      The content following the underscore (_) is the sequence index of a placeholder in the network script. Placeholders are arranged in alphabetical order of the index. If the number of placeholders is greater than 10, the sequence is xxx_0 -> xxx_10 -> xxx_2 -> xxx_3. In the network script, the placeholder with index 10 is placed before the placeholder with index 2. As a result, the defined shape range does not match the input placeholder.

      To avoid this problem, when the number of input placeholders is greater than 10, you are advised to specify the placeholder names in the network script. In this case, the placeholders are named based on the specified names, to associate the shape ranges with the placeholder names.

    - This option cannot be used together with dynamic_dims. If both are configured, dynamic_dims takes precedence and this option is ignored.

## graph_memory_max_size

Sizes of the network static memory and the maximum dynamic memory (used in earlier versions).

In the current version, this parameter does not take effect. The system dynamically allocates memory resources based on the actual memory usage of the network.

## variable_memory_max_size

Size of the variable memory (used in earlier versions).

In the current version, this parameter does not take effect. The system dynamically allocates memory resources based on the actual memory usage of the network.
