# Pre-tuning Check

## Checklist

Before you start accuracy debugging, use the following checklist to exclude errors with the benchmark model or model porting process.

- Pre-debugging checklist

  | Item | Description | Result |
  | --- | --- | --- |
  | [Constant Validation Accuracy](#constant-validation-accuracy) | The benchmark model should be able to offer constant predictions. If the benchmark model does not meet this requirement, it is not competent to offer the accuracy benchmark. | Passed/Failed/Not checked |
  | [Mixed Precision Training](#mixed-precision-training) | As Ascend AI Processor (or NPU) hardware architecture supports only mixed precision training for the user model, the user model needs to be trained with mixed precision. If mixed precision training is not enabled or not enabled successfully for the user model, the NPU may fail to train the model or the accuracy of the trained model may not meet the expectation. | Passed/Failed/Not checked |

- Ported script check

  | Item | Description | Result |
  | --- | --- | --- |
  | [Mixed Precision Training on the NPU](#mixed-precision-training-on-the-npu) | Before accuracy debugging, ensure that the model is successfully migrated to the NPU. Ensure that distributed training (if involved) is enabled, and mixed precision training is enabled during model migration. | Passed/Failed/Not checked |
  | [Loss Scaling on the NPU](#loss-scaling-on-the-npu) | Loss scaling must be enabled in the script migrated to the NPU. Generally, the LossScaleManager parameters need to be configured, as the NPU differs from the GPU in mixed precision computing. | Passed/Failed/Not checked |
  | [Dataset Processing](#dataset-processing) | Check the dataset integrity. The training dataset is always large and easily to get incomplete. | Passed/Failed/Not checked |
  | [Data Preprocessing](#data-preprocessing) | The data preprocessing part of your code may have an automatically-set resource-based variable, which will lead to different dataset shuffle orders. Check the API calls related to data preprocessing in the code to minimize the difference. | Passed/Failed/Not checked |
  | [Shard Method](#shard-method) | The data preprocessing part of the user model code may shard datasets to different nodes based on file name or number of files. This results in large differences between the user model and benchmark model or even files sharded repeatedly to a single node, as the file read API sorts file names differently on different nodes. Add debugging code to exclude such problems, ensuring the sharding policy consistent with that of the benchmark model. | Passed/Failed/Not checked |
  | [Training Procedure](#training-procedure) | During training, a process error such as not clearing intermediate activations occurs frequently, which causes accuracy difference from the benchmark model. Get familiar with the training process and check your training and validation steps. | Passed/Failed/Not checked |
  | [Model Hyperparameters](#model-hyperparameters) | The hyperparameters set in the ported script differ from those set in the benchmark model. Ensure that the hyperparameters in use are the same as those set in the benchmark model. | Passed/Failed/Not checked |

## Benchmark Model Script Check

### Constant Validation Accuracy

A good benchmark model should make the same prediction despite how many times the training script is executed.

> [!NOTE]NOTE
> Unless otherwise specified, randomly-shuffled datasets are the prerequisite for training more than once.

If the benchmark model does not meet the standard, check if the following conditions that help avoid large differences between predictions are true:

- The model has a stable algorithm.
- The dataset quality is high.
- The hyperparameters are stable.

**For a well-developed model \(using hyperparameter borrowing\):**

1. Check the hyperparameters in use and ensure that they are consistent with the given benchmarks.
2. For cluster training, check that the cluster training mode is the same as the given benchmark.
3. Check that dataset files are the same as the given benchmarks.
4. Check the model code and parameters and ensure that the compute logic is consistent with the given benchmark.
5. Check the computational graph and ensure that the computation process and operator shapes are consistent with the given benchmarks.
6. Retrain the model and validate the accuracy of the retrained model. If the accuracy is still different from the benchmark accuracy, repeat the preceding steps until it reaches the benchmark accuracy.
7. Perform training more than three times and check that the validation accuracy of each training is the same as the benchmark accuracy. Repeat the preceding steps until all the preceding conditions are met.

**For user-defined hyperparameters for a well-developed model:**

1. Check that labels are correct in both information and format if the model is trained on a custom dataset.
2. Use the dataset matched with the well-developed model directly \(or tailoring it to your needs\) if your dataset's quality is not guaranteed.
3. Generate a set of candidate hyperparameters by tailoring the dataset matched with the well-developed model based on the training dataset in use and the cluster scale.
4. Perform training more than three times and check that the validation accuracy is constant. Repeat the preceding steps until all the preceding conditions are met.

**For a custom model:**

- Check that the dataset samples are correctly labeled.
- Select a group of stable candidate hyperparameters after debugging.
- Perform training more than three times and check that the validation accuracy is constant. Adjust the hyperparameters and model structure, and repeat the preceding steps until all the preceding conditions are met.

### Mixed Precision Training

If the benchmark model is trained in float32 on the GPU or CPU, the prediction does not change in mixed precision training \(using float16 on the GPU\).

If the benchmark model is trained in mixed precision mode \(in float16 on the GPU\), the prediction does not change in float32 training on the GPU or CPU.

> [!NOTE]NOTE
> If the result of training in high-precision mode fails to be obtained due to memory or hardware limitations, ensure that a constant model accuracy is achieved.

As  Ascend AI Processor  \(or NPU\) hardware architecture supports only mixed precision training, the user model must be trained on the GPU with mixed precision to obtain a converged benchmark model. If the user model is not validated to reach convergence after mixed precision training on the GPU, the model may fail to converge after being ported to the NPU.

The model is not qualified for a benchmark if the result of the comparison of mixed precision training and high-precision training does not meet the accuracy analysis requirements, as this means that mixed precision training in float16 significantly perturbs the model accuracy. In this case, adjust the model structure to avoid accuracy risk in mixed precision training on both the GPU and NPU.

The reference steps are as follows:

1. Check that training with mixed precision is enabled for the benchmark model.
2. Check that dynamic loss scaling is enabled.

    You can also enable static loss scaling, but it is not recommended, as you need to adjust the loss scale value on the NPU to avoid frequent overflow or underflow when using the loss scale value set on the GPU.

3. Check the proportion of floating-point exceptions reported during mixed precision training. A percentage less than 0.5% \(0.1% for a large global batch size\) is recommended.
4. Modify the initial value and scale factor of the loss scale to minimize the number of floating-point exceptions.
5. Perform training more than three times and check that the validation accuracy is constant. Adjust the hyperparameters and model structure, and repeat the preceding steps until all the preceding conditions are met.

## Ported Script Check

### Mixed Precision Training on the NPU

The model is successfully ported to the NPU before accuracy debugging. Distributed training \(if involved\) is enabled.

Especially, mixed precision training is enabled during model porting.

There are two methods to enable mixed precision training on the GPU:

- [Manual mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#training_tensorflow)  on the GPU: All operator data types have been defined in the model.
- [Automatic mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorflow-amp)  on the GPU: Operator data types are defined using TensorFlow or other third-party APIs \(such as apex APIs\).

Note that you should enable only one of the preceding methods to avoid unexpected problems invited by frequent graph modification. In addition, the same method should be used in case of porting to the NPU and the NPU accuracy configuration is as follows:

For the  Ascend 910_95 AI Processor, use the  **precision_mode_v2**  parameter. The value is  **origin**.

For the  Atlas A3 training products/Atlas A3 inference products, use the  **precision_mode_v2**  parameter. The value is  **origin**.

For the  Atlas A2 training products/Atlas A2 inference products, use the  **precision_mode_v2**  parameter. The value is  **origin**.

For the  Atlas training products, use the  **precision_mode**  parameter. The value is  **allow_fp32_to_fp16**.

### Loss Scaling on the NPU

In mixed precision computing, the narrower dynamic range of float16 leads to floating-point overflow/underflow during gradient calculation as well as parameter update failure. Loss scaling can prevent the divergence during mixed precision training.

Loss scaling refers to multiplying the resultant loss in the forward pass by a loss scale  **S**  prior to backpropagation, to avoid gradient values from becoming unrepresentable in float16. After the parameter gradient aggregation and before the optimizer updates parameters, the aggregated parameter gradient is multiplied by 1/**S**.

Dynamic loss scaling checks the gradient floating-point compute exceptions during training and selects the loss scale  **S**  adaptively with the gradient change in the training process.

In practice, as floating-point compute on the  Ascend AI Processor  is different from that on the GPU, floating-point exception detection may show different results. As such, you need to properly configure loss scaling on the NPU.

Reference steps: [Replacing LossScaleOptimizer](../script_migration/manual_porting.md#replacing-lossscaleoptimizer)

### Dataset Processing

Your dataset is the same as the dataset matching the benchmark model.

Generally, datasets are large and easily to produce an incomplete copy. You can use a checksum to check dataset integrity. The operation steps are as follows:

1. Check that the dataset file list of the ported model is the same as that of the benchmark model.
2. Check that the MD5 checksums of dataset files for the benchmark model and ported model are the same.

### Data Preprocessing

The data preprocessing part of your code is the same as that in the benchmark model script.

The data preprocessing part of your code may have an automatically-set resource-based variable, which will lead to different dataset shuffle orders. Check the API calls related to data preprocessing in the code to minimize the difference. The following gives a typical example:

When shuffling dataset, the buffer size is set on top of automatic host memory size query. If the NPU host memory size is greatly different from that of the benchmark model host, dataset shuffle orders will also be different significantly, resulting in unsatisfactory model accuracy.

The operation steps are as follows:

1. Check that the number of files read via the file read API is the same as that of the benchmark.
2. Check that the method in which the source data is converted into the input samples is unchanged.
3. Check that the samples are padded in the same way as the samples input into the benchmark model.
4. Check that the samples are shuffled in the same way \(such as with the same number of samples in the batch to shuffle, or the parallelism during dataset shuffling\) as those of the benchmark.

### Shard Method

The datasets are sharded in the same way as the benchmark model.

The data preprocessing part of the user model code may shard datasets to different nodes based on file name or number of files.

This results in large differences between the user model and benchmark model or even files sharded repeatedly to a single node, as the file read API sorts file names differently on different nodes.

Add debugging code to exclude such problems, ensuring the sharding policy consistent with that of the benchmark model. The reference steps are as follows:

1. Print the lists of files input into the benchmark model and the ported model.
2. Check that the dataset files are sharded to the nodes in the same way.

### Training Procedure

The following are not changed after the model is ported: the initial status, intermediate steps, and results of the training, as well as the samples to validate on and validation process.

During training, a process error such as not clearing intermediate activations occurs frequently, which causes accuracy difference from the benchmark model.

Get familiar with the training process and check your training and validation steps. The reference steps are as follows:

1. Check the weight initialization mode. When using random initialization for the weights, ensure that the randomness is consistent with the benchmark. When initializing weights by loading a pre-trained weight file, ensure that the weight file is consistent with the benchmark.
2. Check the startup script and parameters set in it.
3. Ensure that the distributed training is correctly configured. In particular, avoid the common issue where each node performs training independently without any information synchronization.

### Model Hyperparameters

The hyperparameter values in use are identical to those set in the benchmark model.

The hyperparameters set in the ported script may differ from those set in the benchmark model.

Ensure that the hyperparameters in use are the same as those set in the benchmark model.

Common issues include:

- During model porting of distributed training, the global batch size is incorrectly calculated based on the single-device batch size. As a result, the global batch size on the NPU is different from that of the benchmark model.
- Similar problems happen to the global learning rate.

The reference steps are as follows:

1. Review the hyperparameters in use.
2. Check the hyperparameter configuration files.
3. Compare the hyperparameters by debugging the benchmark model script and the ported script or printing the hyperparameter values.
4. Compare the learning rates in use.
