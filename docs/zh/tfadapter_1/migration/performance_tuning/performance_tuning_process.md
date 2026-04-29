# 性能调优流程

迁移到AI处理器进行训练的网络如果存在性能不达标的问题，可按照如下流程进行性能调优：

**图 1**  性能调优流程
![](../figures/performance_tune_process.png "性能调优流程")

1. 当性能不达标时，优先推荐进行如下通用的性能提升操作。

    1. 使能自动混合精度训练模式。
    2. 检查亲和接口是否替换。
    3. 使能训练迭代循环下沉。
    4. 使用AOE工具进行子图、算子以及梯度切分策略的调优。

        > [!NOTE]说明
        > Ascend 950PR/Ascend 950DT不支持AOE工具。

    详细操作请参见[基本调优](basic_tuning.md)。

2. 再次执行模型训练，并评估训练性能是否达标。
    - 若性能达标 —\> 调优结束。
    - 若性能不达标 —\> 执行步骤3。

3. 使用Profiling工具采集性能数据并分析。

    参考[Profiling数据采集与分析](Profile_data_collection_and_analysis.md)进行性能数据的采集、解析导出与分析。

4. 根据识别的性能瓶颈参见[进阶调优](advanced_tuning.md)进行进一步的性能提升。
5. 再次执行模型训练，进行回归测试，评估训练性能是否达标。
    - 若性能达标 —\> 调优结束。
    - 针对如下产品，若性能不达标 —\> 再次执行[AOE自动调优](automatic_aoe_tuning.md)。

        Atlas A3 训练系列产品/Atlas A3 推理系列产品

        Atlas A2 训练系列产品/Atlas A2 推理系列产品

        Atlas 训练系列产品

    - 针对Ascend 950PR/Ascend 950DT，若性能不达标，请再次执行[Profiling数据采集与分析](Profile_data_collection_and_analysis.md)。
