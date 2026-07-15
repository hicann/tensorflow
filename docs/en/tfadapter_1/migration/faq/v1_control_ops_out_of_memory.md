# What Do I Do If the V1 Control Flow Operator Causes Insufficient Memory?

## Symptom

An error message is reported during model execution, indicating that the memory usage exceeds 31 GB.

![](../figures/tf_v1_faq.png)

## Possible Cause

The graph of the network contains the V1 control structure "switch -\> merge". The network has a great number of branches and uses V1 control flow operators, which affects memory overcommitment and results in memory insufficiency.

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
