# 试验参数

试验参数为调试功能扩展参数，后续版本可能会存在变更，不支持应用于商用产品中。

## experimental_config

功能扩展参数，当前暂不建议使用。用户在创建NPURunConfig之前，可以实例化一个ExperimentalConfig类进行功能配置。ExperimentalConfig类的构造函数，请参见[ExperimentalConfig构造函数](../experimentalconfig_constructor.md)。

## jit_compile

模型编译时，选择是优先在线编译算子，还是优先使用已编译好的算子二进制文件。

- auto（默认值）：针对静态shape网络，在线编译算子；针对动态shape网络，优先查找系统中已编译好的算子二进制，如果查找不到对应的二进制，再编译算子。
- true：在线编译算子，系统根据得到的图信息进行融合及优化，从而编译出运行性能更优的算子。
- false：优先查找系统中已编译好的算子二进制文件，如果能查找到，则不再编译算子，编译性能更优；如果查找不到，则再编译算子。

> [!NOTE]说明
> 该参数仅限于大型推荐类型网络使用。

配置示例：

```python
config = NPURunConfig(jit_compile="auto")
```

## shape_generalization_mode

当“jit_compile”参数配置为“true”（即在线编译算子的场景）时，可通过此参数配置输入shape的泛化模式。

- STRICT（默认值）：直接使用当前迭代的shape，不进行泛化。
- FULL：若两次迭代之间的shape发生变化，则将所有轴的shape泛化为-1。
- ADAPTIVE：若两次迭代之间的shape发生变化，仅将发生变化的轴的shape泛化为-1。新增泛化的轴会触发模型重新编译，因此该配置下模型可能需要多次编译。

> [!NOTE]说明
> 当compile_dynamic_mode配置为True时，首次迭代会将所有输入shape泛化为“-1”，此时shape_generalization_mode的配置将不生效。

配置示例：

```python
config = NPURunConfig(shape_generalization_mode="FULL")
```

## auto_multistream_parallel_mode

该参数仅适用于静态shape图场景，开发者可通过配置此参数开启Cube算子与Vector算子的并行执行，以提升图执行性能。

- cv：代表开启Cube算子与Vector算子的并行执行功能。
- None（默认值），即不开启Cube算子与Vector算子的并行执行功能。

> [!NOTE]说明
>
>- 该参数仅限于推荐类型网络的训练场景使用。
>- Cube算子与Vector算子的并行执行功能不可以与多流并发执行功能（通过环境变量ENABLE_DYNAMIC_SHAPE_MULTI_STREAM设置）同时启用。
> 关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。

配置示例：

```python
config = NPURunConfig(auto_multistream_parallel_mode="cv")
```
