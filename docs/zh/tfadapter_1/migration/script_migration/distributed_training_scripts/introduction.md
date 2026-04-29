# 简介

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式训练可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。分布式训练通过将计算任务按照一定的方法拆分到不同的AI处理器上来加速模型的训练速度，拆分的计算任务之间通过集合通信来完成信息的汇总和交换，完成整个训练任务的并行处理，从而实现加快计算任务的目的。

## 使用前须知

参考本章节进行分布式训练脚本迁移前，首先需要参见[单机单卡脚本迁移](../single_service_single_device_porting.md)完成数据预处理、模型构建、训练执行等基本流程的代码适配。

## TF Adapter支持的分布式API

在TensorFlow中，一般使用tf.distribute.Strategy进行分布式训练，具体请参考[https://www.tensorflow.org/guide/distributed_training](https://www.tensorflow.org/guide/distributed_training)。而AI处理器暂不支持上述分布式策略，因此对于原始TensorFlow训练脚本，需要经过修改后，才可以在NPU上支持分布式训练。

TF Adapter支持的主要分布式接口如下：

- [npu_distributed_optimizer_wrapper](../../../apiref/npu_optimizer/npu_distributed_optimizer_wrapper.md)：将TensorFlow原生梯度训练优化器和NPU的allreduce操作合并为一个函数，用于实现各个Device之间计算梯度、梯度聚合操作。
- [npu_allreduce](../../../apiref/npu_optimizer/npu_allreduce.md)：用于原始TensorFlow脚本使用了梯度计算接口的场景（例如tf.gradients），梯度计算完成后，需调用此接口对梯度进行allreduce和梯度更新。
- 通信域管理接口：包括create_group、destroy_group、get_rank_size、get_rank_id等接口，详细可参见《[HCCL通信域管理接口（Python语言）](https://hiascend.com/document/redirect/CannCommunityHcclPythonApi)》。
- 集合通信接口：包括allreduce、allgather、broadcast、reduce_scatter、reduce、alltoallv等接口，详细可参见npu_bridge.hccl.hccl_ops。
