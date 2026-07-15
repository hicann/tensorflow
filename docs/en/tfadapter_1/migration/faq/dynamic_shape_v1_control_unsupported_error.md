# What Do I Do If an Error Message Indicating Unsupported V1 Control Flow Operators Is Displayed During Dynamic-Shape Network Execution?

## Symptom

The following error is reported during model execution:

```text
node node_name(node_type) is v1 control operator, which is not supported, please convert to v2 control operator
```

## Possible Cause

The current network is a dynamic-shape network and contains control flow operators of TensorFlow V1. However, the dynamic-shape network does not support V1 control flow operators. As a result, the network fails to be executed.

## Solution

Convert the control flow operators of TensorFlow V1 on the network to those of TensorFlow V2.

- Method 1 \(recommended\): Modify the network script to add the following two instructions after  **import tensorflow as tf**.

    ```python
    tf.enable_control_flow_v2()
    tf.enable_resource_variables()
    ```

- Method 2: Configure the environment variable to change the control flow operators of V1 to those of V2.

    ```bash
    export ENABLE_FORCE_V2_CONTROL=1
    ```

    Note: If this environment variable is used, the control flow operators of V1 may fail to be converted to those of V2, for example, the ref control operator contained in the network script.
