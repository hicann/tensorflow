# Accuracy Debugging Process

This section describes how to tune the accuracy when the ported model functions properly on the  AI processor  but still suffers from accuracy issues or poor convergence.

## Background

After the ported model is trained on  AI processor  \(NPU for short\) and functions properly, the accuracy may not meet the requirements or the convergence effect may be poor. When the model runs on  AI processor, the issues that may be encountered include but are not limited to the following:

- The loss curve differs greatly from that of the benchmark model.
- The validation accuracy differs greatly from that of the benchmark model.

These accuracy issues are difficult to locate due to the following reasons:

- The training is completed without exceptions.
- No warning or error is recorded in the logs.
- The differences are found only during comparison with the benchmark model.

This section provides guidance for you to tune the accuracy.

## Precision Debugging Approach

Accuracy problems result from various aspects, for example, the provided benchmark model, an error occurred during model porting, or operator accuracy on the network. The following flowchart summarizes the workflow for accuracy tuning with the possible causes highlighted.

![](../figures/accuracy_tuning_workflow.png)

The following table describes the accuracy debugging process.

| Step | Description |
| --- | --- |
| [Pre-tuning Check](pre-tuning_check.md) | Check the following items before accuracy tuning:<br>  - Unported script: Check that the benchmark model is qualified.<br>  - Ported script: Check that no errors occur during model porting. |
| [Model Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md) | Before accuracy tuning, install One-Click Accuracy Analyzer on your training NPU. |
| [Floating-Point Exception Detection](floating-point_exception_detection.md) | At network run time, floating-point exceptions happen from time to time. That is, the loss scale decreases many times or directly to 1. In this case, analyze the overflow and underflow data to determine the problem source. |
| [Fusion Exception Detection](fusion_exception_detection.md) | At network run time, the system fuses operators according to built-in fusion patterns for better network performance. As most fusions are processed automatically, it is possible that your model contains an operator that is not yet covered by the fusion implementations, which impacts model accuracy. You can disable fusion to determine whether the problem happens in operator fusion phase. |
| [Network-wide Accuracy Comparison](network_accuracy_comparison.md) | If the accuracy still does not meet expectations after the above steps, collect operator execution results (dump data) during training and compare them with results from the benchmark operator (such as TensorFlow). This helps quickly pinpoint operators with accuracy issues. |
| [Random Error Detection](random_error_detection.md) | At network run time, the calculation with the same inputs may produce different outputs. If such random errors happen, you can perform training twice, collect the compute result (that is, dump data) of each operator, and compare the data to quickly locate the fishy operator layer that causes the errors. |
