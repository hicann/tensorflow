# model_to_npu_estimator

## 功能说明

将通过Keras构建的模型转换为NPUEstimator对象。

## 函数原型

```python
def model_to_npu_estimator(keras_model=None,
                           keras_model_path=None,
                           custom_objects=None,
                           model_dir=None,
                           checkpoint_format='saver',
                           config=None,
                           job_start_file='')
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| keras_model | 已经编译好的Keras模型对象。<br>该参数与keras_model_path不可同时传入。 |
| keras_model_path | 保存在磁盘上的已编译Keras模型的路径。可以使用Keras模型的save()方法生成HDF5格式的Keras模型。<br>该参数与keras_model不可同时传入。 |
| custom_objects | 自定义对象的字典，在构造Keras时，如果有自定义的层或者函数，在加载模型时需要使用custom_objects。 |
| model_dir | 保存模型路径，用于保存或恢复模型文件。如果没有配置，那么将使用config中的model_dir配置。如果都设置了，那这两个配置项必须一样。如果都设置为None，就会使用临时的文件夹/tmp。 |
| checkpoint_format | 设置训练时NPUEstimator保存的checkpoint的格式。取值：<br><br>  - saver（默认）：表示通过tf.train.Saver()保存模型。<br>  - checkpoint：表示通过tf.train.Checkpoint ()保存模型，tf.train.Checkpoint与tf.train.Saver相比，强大之处在于其支持在即时执行模式下“延迟”恢复变量。 |
| config | NPURunConfig类对象，用于配置NPUEstimator的运行参数。<br>关于NPURunConfig类的构造函数，请参见[NPURunConfig构造函数](../npu_config/npurunconfig_constructor/README.md)。 |
| job_start_file | CSA场景下用于启动训练进程的配置文件路径。 |

## 返回值

根据传入的keras model返回一个NPUEstimator对象。

## 约束说明

目前仅功能模型和序列模型（为Keras构图方式）支持通过model_to_npu_estimator接口转换为NPUEstimator对象。
