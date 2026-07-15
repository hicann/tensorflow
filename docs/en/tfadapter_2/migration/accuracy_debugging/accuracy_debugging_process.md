# Accuracy Tuning Process

## Background

After the ported model is trained on  AI processor  \(NPU for short\) and functions properly, the accuracy may not meet the requirements or the convergence effect may be poor. When the model runs on  AI processor, the issues that may be encountered include but are not limited to the following:

- The loss curve differs greatly from that of the benchmark model.
- The validation accuracy differs greatly from that of the benchmark model.

These accuracy issues are difficult to locate due to the following reasons:

- The training is completed without exceptions.
- No warning or error is recorded in the logs.
- The differences are found only during comparison with the benchmark model.

This section provides guidance for you to tune the accuracy.

## Tuning Approach

The possible causes of accuracy issues are as follows:

1. Bad benchmark model
2. Improper model porting
3. Operator accuracy errors

The following flowchart summarizes the workflow for accuracy tuning with the possible causes highlighted.

![](../figures/accuracy_tuning_workflow.png)

The following table describes the accuracy debugging process.

| No. | Step | Description |
| --- | --- | --- |
| 1 | [Pre-tuning Check](pre-tuning_check.md) | Check the following items before accuracy tuning:<br>  - Unported script: Check that the benchmark model is qualified.<br>  - Ported script: Check that no errors occur during model porting. |
| 2 | [One-Click Accuracy Analyzer Deployment](./accuracy_analyzer_deployment.md) | Before accuracy tuning, install one-click accuracy analyzer on your training NPU. |
| 3 | [Floating-Point Exception Detection](floating-point_exception_detection.md) | At network run time, floating-point exceptions happen from time to time. That is, the loss scale decreases many times or directly to 1. In this case, analyze the overflow and underflow data to determine the problem source. |
| 4 | [Fusion Exception Detection](fusion_exception_detection.md) | At network run time, the system fuses operators according to built-in fusion patterns for better network performance. As most fusions are proceeded automatically, it is possible that your model contains an operator that is not yet covered by the fusion implementations, which impacts model accuracy. You can disable fusion to determine whether the problem happens in operator fusion phase. |
| 5 | [Network Accuracy Comparison](network_accuracy_comparison.md) | If the accuracy problem does not happen in the steps above, dump the compute result of each operator during the training process and compare the dump data with that of each benchmark operator (such as the TensorFlow equivalents) to quickly spot the faulty operators. |
| 6 | [Random Error Detection](random_error_detection.md) | At network run time, the calculation with the same inputs may produce different outputs. If such random errors happen, you can perform training twice, collect the compute result (that is, dump data) of each operator, and compare the data to quickly locate the fishy operator layer that causes the errors. |
