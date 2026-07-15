# model_to_npu_estimator

## Description

Converts the model constructed by using  **Keras**  to an  **NPUEstimator**  object.

## Prototype

```python
def model_to_npu_estimator(keras_model=None,
                           keras_model_path=None,
                           custom_objects=None,
                           model_dir=None,
                           checkpoint_format='saver',
                           config=None,
                           job_start_file='')
```

## Parameters

| Parameter | Description |
| --- | --- |
| keras_model | Built Keras model object.<br>This parameter and keras_model_path cannot be input at the same time. |
| keras_model_path | Path for storing the built Keras model on the drive. You can use the save() method of the Keras model to generate a Keras model in HDF5 format.<br>This parameter and keras_model cannot be input at the same time. |
| custom_objects | Dictionary of user-defined objects. If a user-defined layer or function is used during Keras construction, custom_objects must be used during model loading. |
| model_dir | Model directory, which is used to save or restore model files. If this parameter is not set, the value of model_dir in the config file is used. If both parameters are set, the values of the two parameters must be the same. If both parameters are set to None, the temporary directory /tmp is used. |
| checkpoint_format | Sets the format of the checkpoint file saved by NPUEstimator during training.<br><br>  - saver (default): Save the model using tf.train.Saver().<br>  - checkpoint: Save the model using tf.train.Checkpoint (). Compared with tf.train.Saver, tf.train.Checkpoint supports delayed variable recovery in instant execution mode. |
| config | NPURunConfig class object, which is used to configure NPUEstimator running parameters.<br>For details about the constructor of the NPURunConfig class, see [NPURunConfig Constructor](../npu_config/npurunconfig_constructor/README.md). |
| job_start_file | Path of the configuration file used to start the training process in the CSA scenario. |

## Returns

An  **NPUEstimator**  object is returned based on the input Keras model.

## Restrictions

Currently, only the function model and sequence model \(Keras graph construction mode\) can be converted into an NPUEstimator object using the  **model_to_npu_estimator**  API.
