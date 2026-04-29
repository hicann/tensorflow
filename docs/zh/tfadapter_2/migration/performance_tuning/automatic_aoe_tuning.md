# AOE自动调优

AOE自动调优工具通过生成调优策略、编译、在运行环境上验证的闭环反馈机制，不断迭代出更优的调优策略，最终得到最佳的调优策略，从而可以更充分利用硬件资源，提升网络的性能。模型训练阶段，分别使能AOE工具进行子图/算子与梯度切分的调优，调优完成后，最佳调优策略会固化到知识库。模型再次训练时，无需开启调优，即可享受知识库带来的性能收益。

> [!CAUTION]注意
> AOE调优特性仅支持如下产品：
>
> - Atlas A3 训练系列产品/Atlas A3 推理系列产品
> - Atlas A2 训练系列产品/Atlas A2 推理系列产品
> - Atlas 训练系列产品

建议按照如下调优顺序使用AOE工具进行调优：

![](../figures/aoe_tune.png)

> [!NOTE]说明
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，Atlas A3 训练系列产品/Atlas A3 推理系列产品，不支持子图调优。

训练场景下使能AOE调优有两种方式：

- 设置环境变量

    ```bash
     # 1：子图调优  2：算子调优   4：梯度调优
    export AOE_MODE=2
    ```

- 修改训练脚本，在初始化NPU设备前通过添加[AOE](../../apiref/npu-global_options/AOE.md)中的“aoe_mode”参数指定调优模式：

    ```python
    import npu_device as npu
    npu.global_options().aoe_config.aoe_mode="2"
    npu.open().as_default()
    ```
