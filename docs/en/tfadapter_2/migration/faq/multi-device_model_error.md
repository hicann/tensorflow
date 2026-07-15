# How Do I Fix Application Errors Caused by Model Execution on Multiple Devices?

## Symptom

In distributed training or inference scenarios, the process stops responding when the service script is executed.

Search for "The model has been compiled on the Ascend AI processor, current graph id is" in logs. Assume that there are eight devices. The following information is displayed:

![](../figures/multidevice_error.png)

Under normal conditions, there should be multiple instances of the same graph ID on each device, with each ID appearing the same number of times. According to the preceding logs, there are eight graphs with ID 61 and eight graphs with ID 71, which is a normal case. However, there are only four graphs with ID 81. The graph counts are different across devices.

## Possible Cause

When a task is executed on multiple devices, the models executed on the devices are inconsistent, which triggers model recompilation on some devices. As a result, the IDs of the HCCL operators are inconsistent, the communication function is abnormal, and the process is suspended.

The principle of triggering model recompilation on a TensorFlow network is as follows:

When a user script calls the  **session.run**  API, it searches for or creates the corresponding graph executor. If the graph changes, the corresponding executor cannot be hit from the cache of the previous graph compilation. In this case, TensorFlow creates a new executor and re-compiles the graph. The recompilation causes inconsistent HCCL operator IDs, and the HCCL communication fails.

## Solution

Check the service script, find out the model execution differences on different devices, and modify the script.

Common methods \(for reference only\):

- Check whether the script contains logic similar to the "if \(xxx % rank_id\) == n" format. This logic may cause different graph structures on different devices.
- Check whether  **tf.summary**  is enabled on some devices.  **tf.summary**  causes model recompilation.
