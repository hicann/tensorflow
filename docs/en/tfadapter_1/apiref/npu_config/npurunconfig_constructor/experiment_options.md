# Experiment Parameters

The experiment parameters are extended parameters for debugging and may be changed in later versions. Therefore, they cannot be used in production environments.

## experimental_config

Extended parameter. Currently, this parameter is not recommended. Before creating NPURunConfig, you can instantiate an ExperimentalConfig class to configure functions. For details about the constructor of the ExperimentalConfig class, see [ExperimentalConfig Constructor](../experimentalconfig_constructor.md).

## jit_compile

Determines whether to compile the operator online or use the compiled operator binary file.

- auto: For a static shape network, compile the operator online. For a dynamic shape network, search for the compiled operator binary in the system first. If the corresponding binary file is not available, compile the operator.
- true: Operators are compiled online. The system performs fusion and tuning based on the obtained graph information to get better performing operators.
- false: The compiled operator binary file in the system is preferentially searched. If the file can be found, operators are not compiled anymore, which produces better compilation performance. If the file cannot be found, operators will be compiled.

> [!NOTE]
> This option is used only for networks of large recommendation models.

Example:

```python
config = NPURunConfig(jit_compile="auto")
```

## shape_generalization_mode

When jit_compile is set to true (online operator compilation), use this parameter to configure the shape generalization mode.

- STRICT (default): Uses the shape of the current iteration as is, without any generalization.
- FULL: Generalizes all axes to -1 if the shape changes between iterations.
- ADAPTIVE: Generalizes only the shape of the changed axis to -1 if the shape changes between iterations. The newly generalized axis triggers model recompilation, which may cause the model to be compiled multiple times under this configuration.

> [!NOTE]
> If compile_dynamic_mode is set to True, all input shapes are generalized to -1 in the first iteration. In this case, the configuration of shape_generalization_mode does not take effect.

Example:

```python
config = NPURunConfig(shape_generalization_mode="FULL")
```

## auto_multistream_parallel_mode

This option applies to static and dynamic shape graph scenarios. You can enable parallel execution of Cube and Vector operators to improve graph execution performance.

- **cv**, Parallel execution of Cube and Vector operators is enabled.
- **LoadBalance:n**, Load balancing algorithm that distributes all operators evenly across n streams for execution. Here, n represents the maximum number of streams, which must be a positive integer within the range [1, 64]. If n exceeds the number of available cores, performance may degrade.
- **MainStream:n**，Main stream algorithm that executes serial operators on the main stream, while other parallelizable operators are distributed across other streams. Here, n represents the maximum number of streams, which must be a positive integer within the range [1, 64]. If n exceeds the number of available cores, performance may degrade.
- The default value is empty, meaning Cube and Vector operators are executed serially.

> [!NOTE]NOTE
>
> - This option is used only for recommendation networks.
> - To use this function in dynamic shape multi-stream mode, you need to first enable dynamic shape multi-stream by setting the environment variable ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, and then configure this option. For details about environment variables, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Example:

```python
config = NPURunConfig(auto_multistream_parallel_mode="cv")
```
