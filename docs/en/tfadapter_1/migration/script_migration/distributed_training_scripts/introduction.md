# Overview

In deep learning, as the numbers of datasets and parameters increase, so do the time and hardware resources required for training, which brings a potential bottleneck. Distributed training is a popular optimization technique for training, which has lower requirements on hardware resources such as memory and compute performance. In distributed training, a training job is partitioned and distributed across  AI processors for improved training efficiency. Training jobs exchange and summarize information via collective communication.

## Precautions

Before porting distributed training scripts, complete code adaptation for basic processes such as data preprocessing, model build, and training execution by referring to  [Using Single-Server Single-Device Scripts](../single_service_single_device_porting.md).

## Distributed APIs Supported by TF Adapter

In TensorFlow,  **tf.distribute.Strategy**  is generally used for distributed training. For details, visit  [https://www.tensorflow.org/guide/distributed_training](https://www.tensorflow.org/guide/distributed_training). Currently, the  AI processor  does not support the preceding distributed policies. Therefore, the original TensorFlow training script needs to be modified to support distributed training on NPUs.

TF Adapter supports the following distributed APIs:

- [npu_distributed_optimizer_wrapper](../../../apiref/npu_optimizer/npu_distributed_optimizer_wrapper.md): Combines the native TensorFlow's gradient training optimizer and NPU AllReduce operations into one function to implement gradient calculation and aggregation between devices.
- [npu_allreduce](../../../apiref/npu_optimizer/npu_allreduce.md): Performs AllReduce and update operations on gradients after gradient calculation is complete in the scenario where the original TensorFlow script uses the gradient calculation API, for example,  **tf.gradients**.
- Communicator management APIs: Include  **create_group**,  **destroy_group**,  **get_rank_size**, and  **get_rank_id**. For details, see  APIs \> HCCL APIs \> Communicator Management \> Python APIs  in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/canncommercial/900/API/hcclug/hcclug_000001.html).
- Collective communication APIs: Include  **allreduce**,  **allgather**,  **broadcast**,  **reduce_scatter**,  **reduce**,  **alltoallv**.
