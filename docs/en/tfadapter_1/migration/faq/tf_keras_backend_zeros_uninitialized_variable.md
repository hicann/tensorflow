# What Do I Do When an Error Indicating Uninitialized Variables Is Reported During Training?

## Symptom

An error is reported during training, indicating that the variable operator is not initialized.

![](../figures/keras-backend-zeros-faq.png)

## Possible Cause

During data preprocessing,  **tf.keras.backend.zeros**  is used to generate variables. The variable operator cannot be executed on the device side and thereby fails to be initialized.

## Solution

Modify the training script. Use the native TensorFlow API  **tf.zeros**  instead of  **tf.keras.backend.zeros**  to generate variables on the host in tensor mode.

Original script

```python
y = {
   'mlm_loss': tf.keras.backend.zeros([1]),
   'mlm_acc': tf.keras.backend.zeros([1]),
}
```

Script after modification

```python
y = {
   'mlm_loss': tf.zeros([1]),
   'mlm_acc': tf.zeros([1]),
}
```
