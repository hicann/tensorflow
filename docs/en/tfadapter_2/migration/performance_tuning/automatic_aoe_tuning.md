# Automatic AOE Tuning

The AOE tool continuously iterates tuning policies through a closed-loop feedback mechanism of policy generation, compilation, and verification in the operating environment, and finally obtains the optimal one. This helps fully utilize hardware resources and improve network performance. During model training, the AOE tool can be enabled to tune subgraphs, operators, and gradients. After the tuning is complete, the optimal tuning policy is added to the repository. When the model is trained again, you can directly use the repository for efficient tuning, without enabling the tuning function.

> [!CAUTION]NOTICE
> The AOE tuning feature supports only the following  Product:
>
> - Atlas A3 training products/Atlas A3 inference products
> - Atlas A2 training products/Atlas A2 inference products
> - Atlas training products

You are advised to use the AOE tool to perform tuning in the following sequence.

![](../figures/aoe_tune.png)

> [!NOTE]NOTE
> For the  Atlas A2 training products/Atlas A2 inference productsAtlas A3 training products/Atlas A3 inference products, subgraph tuning is not supported.

You can enable AOE tuning in training scenarios using either of the following methods:

- Set the environment variable.

    ```bash
     # 1: subgraph tuning; 2: operator tuning; 4: gradient tuning
    export AOE_MODE=2
    ```

- Add  **aoe_mode**  of  [AOE](../../apiref/npu-global_options/AOE.md)  to the training script before initializing the NPU, to specify a tuning mode.

    ```python
    import npu_device as npu
    npu.global_options().aoe_config.aoe_mode="2"
    npu.open().as_default()
    ```
