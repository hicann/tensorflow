# NPU_ENABLE_PERF

## Description

Prints graph time consumption of the TF Adapter in  TensorFlow  2.6.5 training and online inference scenarios.

- **1** or **true**: Prints graph time consumption.
- **0** or **false**: Does not print graph time consumption.

After enabling the TF Adapter graph time consumption printing function, you can search for the keywords  **Graph engine run**  and  **cost**  in logs to query the time consumption. For example:

```bash
Graph engine run 1 times for graph 0 cost 11257 ms
```

## Example

```bash
export NPU_ENABLE_PERF=1
```

## Constraints

- This environment variable takes effect only when the debug logging for TF Adapter is enabled. That is, set  [NPU_DEBUG](NPU_DEBUG.md)  to  **1**  or  **true**.

    ```bash
    export NPU_DEBUG=1
    ```

- This environment variable should be set before the  **import npu_device**  operation.
- This environment variable applies only to the scenario where the  TensorFlow  2.6.5 network training or online inference is performed on the Ascend platform.

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas training product

Atlas inference product
