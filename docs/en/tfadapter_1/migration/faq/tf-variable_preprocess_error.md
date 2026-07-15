# How Do I Fix Training Errors Caused by tf.Variable in Data Preprocessing?

## Symptom

When the TensorFlow-based network is executed, the following error message is displayed:

```text
tensorflow.python.framework.errors_impl.FailedPreconditionError: Error while reading resource variable inference/embed_continuous from Container: localhost.  This could mean that the variable was uninitialized. Not found: Resource localhost/inference/embed_continuous/N10tensorflow3VarE does not exist.
```

## Possible Cause

The  **tf.Variable**  variable exists in the data preprocessing script. When the training script runs on the Ascend platform,  **tf.Variable**  is executed on the host, while the initialization of  **tf.Variable**  is executed on the device. The variable execution and initialization are not performed on the same device. As a result, a training error occurs.

The following is the sample code of a training script that uses  **tf.Variable**:

```python
batch_size = tf.Variable(
    tf.placeholder(tf.int64, [], 'batch_size'),
    trainable= False, collections=[]
)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
```

## Solution

Modify the training script and change  **tf.Variable**  to a constant. The following is an example:

```python
batch_size = 64
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
```
