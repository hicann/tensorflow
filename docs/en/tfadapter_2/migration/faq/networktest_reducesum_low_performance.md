# What Do I Do If Operator ReduceSum Has Poor Performance on a Network?

## Symptom

During network debugging, the overall performance is low. Use the Profiling tool to obtain the profile data of the network and analyze the profile data of the ReduceSum operator. It is found that the performance of the ReduceSum operator does not meet the expectation.

The profiling result of the ReduceSum operator is as follows.

![](../figures/reducesum_profiling.png)

The data type of ReduceSum's input is  **DT_FLOAT16**  and the value of  **block_dim**  is  **1**, which indicates that multiple blocks are not enabled for the operator.

## Solution

For the ReduceSum operator of the  AI processor, if the input data type is float16, multi-core computing cannot be enabled in some scenarios due to hardware restrictions.

Take the ReduceSum operator as an example. If the input data is float16, there are two solutions as follows:

- The mixed precision is not enabled during network debugging and ReduceSum's input is of type float16. In this case, if ReduceSum's performance is poor, you can insert a Cast operator before the ReduceSum operator to cast the data type from float16 to float32.

    Multiple blocks can be enabled when the input data is of the float32 type. As such, the operator performance can be improved.

- The mixed precision is enabled during network debugging and ReduceSum's input data type is cast from float32 to float16. In this case, you can add the ReduceSum operator to the blocklist for mixed precision to avoid the data type being cast to float16 during network debugging, preventing the ReduceSum operator from performance deterioration.

    To add the ReduceSum operator to the blocklist for mixed precision, perform the following steps:

    1. Specify the operator on the blocklist for mixed precision by using  **modify_mixlist**.

        Example:

        ```python
        import npu_device as npu
        npu.global_options().precision_mode = 'allow_mix_precision'
        npu.global_options().modify_mixlist = "/home/test/ops_info.json"
        npu.open().as_default()
        ```

    2. Configure the operator graylist in the  **ops_info.json**  file. The following is a configuration example.

        ```json
        {
            "black-list": {
                "to-add": ["ReduceSumD"]
            }
        }
        ```

        For details, see  [Modifying the Blocklist and Trustlist for Mixed Precision](../performance_tuning/mixed_precision_training.md#modifying-the-blocklist-and-trustlist-for-mixed-precision).

> [!CAUTION]NOTICE
> This solution is dedicated to improving operator ReduceSum's performance under conditions described in  [Symptom](#symptom).
