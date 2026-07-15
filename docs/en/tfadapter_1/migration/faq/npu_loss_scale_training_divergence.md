# What Do I Do If The Trained Model Does Not Converge After the NPU Loss Scale Optimizer Is Used?

## Symptom

The trained word2vec network converges if NPU Loss Scale is disabled. However, after NPU Loss Scale is used, it does not converge.

![](../figures/word2vec_faq.png)

## Possible Cause

After overflow/underflow detection is enabled and Loss Scale logs are printed, it is found that the loss scale value keeps decreasing to 0, which indicates that overflow exists. The analysis on overflow data shows that the Log operator continuously overflows.

The following figure shows the further analysis on dump data.

![](../figures/word2vec_npuloss_scale_faq.png)

As shown in the preceding figure,  **0**  exists in the input of the Log operator. According to the curve of the Log function,  **0**  corresponds to infinity, which indicates that overflow occurs. As a result, dynamic loss scaling cannot be enabled.

After a further analysis on the source of  **0**  in the Log operator, it is found that the Log operator uses the NZ format, which is required in the subsequent MatMul operation. To improve the calculation efficiency and prevent excessive transformation operators on the entire network, the NZ format is spread to the Log operator. Compared to the original format, the NZ format requires  **0**  to be padded to TransData. However,  **0**  does not exist in the data before TransData.

Simply speaking, the Log operator uses the NZ format in mixed precision mode. As a result, the input data is padded with  **0**  by TransData, which leads to overflow during computing. However, the overflowed data is not valid.

## Solution

In the NPU Loss Scale mechanism, detection is enabled by default for overflow that may affect the final gradient result during computing. If a floating-point exception occurs in an iteration, the gradient update in the iteration will be abandoned.

You can use the enable_overflow_check parameter to determine whether to enable overflow/underflow detection.

```python
FixedLossScaleManager(loss_scale=FLAGS.loss_scale, enable_overflow_check=True)
```

- **True**  \(default\): enabled. If overflow is detected in an iteration, parameters of that iteration are not updated.
- **False**: disabled. Parameters are always updated regardless of overflow.

In the word2vec network, the overflow data is not valid data. Even if the floating-point exception is not detected, convergence can still be ensured. Therefore, you can use static loss scaling and disable overflow/underflow detection.

In addition, overflow/underflow detection can be disabled in other scenarios. For example:

- The network script contains overflow that does not affect the computing result, for example, GNMT.
- There is a risk of overflow in the script, but the clip operation is performed subsequently, for example, Faster-RCNN.
- Saturation or overflow occurs only in some networks, which has limited impact on convergence.
