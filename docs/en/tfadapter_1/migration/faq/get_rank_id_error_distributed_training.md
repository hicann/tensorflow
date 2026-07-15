# What Do I Do If "get rank id error" Occurred When Distributed Training Is Performed?

## Symptom

The message "get rank id error" is displayed on the screen, as shown in the following figure:

![](../figures/get_rank_id_error.png)

The error message "Call hcom_bind_model failed" is displayed in the host log, as shown in the following figure:

![](../figures/hcom_bind_failed.png)

## Possible Cause

The Python-based collective communication management APIs can be called only after collective communication is initialized.

## Solution

In the training script, initialize collective communication before calling the Python-based collective communication management APIs. For details about collective communication initialization, see  [Initializing Collective Communication](../others/init_collective_communication.md).
