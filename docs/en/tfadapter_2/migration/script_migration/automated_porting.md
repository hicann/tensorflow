# Automated Porting

## Tool Overview

- Features

  TF Adapter provides a porting tool targeted at TensorFlow 2.6.5. The tool analyzes the support for native TensorFlow Python APIs on the  AI processor, and automatically ports native TensorFlow training scripts to those supported by the  AI processor. The ported script can be trained on  AI processor. For APIs unportable by the tool, modify your training script according to the tool report.

- How to Obtain

  After the CANN software is installed, the porting tool is stored in the  **$\{TFPLUGIN_INSTALL_PATH\}/npu_device/convert_tf2npu/**  directory.

- Constraints

  Before using the tool, you are advised to learn the restrictions on the original training script.

  1. The original script can run on the GPU or CPU successfully for accuracy convergence.
  2. The original script must use only the  [official TensorFlow 2.6 APIs](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf). The Horovod APIs can be used when the script calls the TensorFlow 1._x_  APIs in the form of  **tf.compat.v1**.

     The tool does not support porting if the script uses other third-party APIs. For example:

     1. Native Keras APIs are not supported. However, tf.keras APIs are supported.
     2. CuPy APIs are not supported. The original script using CuPy APIs may fail to run on the  AI processor  even if it can run properly on the GPU.

  3. The TensorFlow module in the original script needs to be referenced as follows. Otherwise, the porting report may not be accurate. \(This does not affect the script porting function.\)

      ```bash
      import tensorflow as tf
      import horovod.tensorflow as hvd
      ```

  4. Data types float64, complex64, complex128, and DT_VARIANT are not supported.
  5. Restrictions on distributed-training script porting:
     1. Before using the tool, you must manually add dataset sharding logic. For details, see "Sharding dataset to workers" in [Distributed Training Script Adaptation (Single Device)](manual_porting.md#distributed-training-script-adaptation-single-device).
     2. Currently, the tool automatically ports only distributed-training scripts that use the TensorFlow Keras optimizer \(including SGD, RMSprop, Adam, Ftrl, Adagrad, Adadelta, Adamax, and Nadam\). Other distributed-training scripts need to be ported manually. For details, see  [Distributed Training Script Adaptation \(Single Device\)](manual_porting.md#distributed-training-script-adaptation-single-device).
     3. For LossScaleOptimizer in the original script, the tool supports only the porting from  **tf.keras.mixed_precision.LossScaleOptimizer**  to  [npu.train.optimizer.NpuLossScaleOptimizer](../../apiref/npu-train-optimizer-npulossscaleoptimizer.md). If your script uses a different LossScaleOptimizer, change it to  **tf.keras.mixed_precision.LossScaleOptimizer**, test the functionality and accuracy, and then manually replace it with  [npu.train.optimizer.NpuLossScaleOptimizer](../../apiref/npu-train-optimizer-npulossscaleoptimizer.md).

  6. The tool cannot automatically enable iteration offload to the NPU at this moment. If iteration offload is used in your original script, manually enable it in the ported script. For details, see  [Setting the Number of Iterations Offloaded to NPU](./manual_porting.md#setting-the-number-of-iterations-offloaded-to-npu).

## Prerequisites

Before porting a model to the  AI processor, prepare a model developed in TensorFlow 2.6.5 and the dataset, and run the model on the GPU or CPU to test its accuracy and performance. In addition, record the accuracy and performance results for later comparison with those on the  AI processor.

## Porting Operation

1. Install dependencies.

    ```bash
    pip3 install pandas==1.3.5
    pip3 install openpyxl
    pip3 install google_pasta
    ```

2. Perform script scanning and automated porting.

    Go to the porting tool directory  **$\{TFPLUGIN_INSTALL_PATH\}/npu_device/convert_tf2npu/**  and run the following command to perform both script scanning and automated porting:

    ```bash
    python3 main.py -i /root/models/examples/test -m /root/models/example/test/test.py
    ```

    **main.py**  is the entry script of the tool. The following table describes the options.

    | Option | Description | Required |
    | --- | --- | --- |
    | -i | Path of the training script to be ported. The path must be a folder.<br>  - The tool scans and ports only the .py files in the folder specified by the -i option.<br>  - You are advised to put all necessary files in the same directory. Otherwise, you need to run the porting command in each directory. | Yes |
    | -o | Path of the ported script. The path cannot be a subdirectory of the original script path.<br>Defaults to the current path, for example, output_npu_20220517172706/xxx_npu_20220517172706. | No |
    | -r | Path of the porting report. The path cannot be a subdirectory of the original script path.<br>Defaults to the current path, for example, report_npu_20220517172706. | No |
    | -m | Python execution entry point file.<br>If the original script does not contain the main function, the tool cannot identify the entry point function, resulting in the failure of NPU resource initialization and NPU training configuration.<br>In that case, you must use -m to specify the entry point file for Python execution, so that the tool can completely port the user script for subsequent training.<br>Example: -m /root/models/xxx.py | No |
    | -d | Distributed strategy of the original script that supports distributed training. Values:<br><br>  - tf_strategy: The original script uses tf.distribute.Strategy.<br>  - horovod: The original script uses the Horovod distributed module. | No |
    | -c | (Or --compat) Whether to use tf.compat.v1 APIs for execution in TensorFlow 1.x mode. | No |

    > [!NOTE]NOTE
    >**python3 main.py -h**: displays the help information of the porting tool.

    - During the porting, check the following information, which indicates related files are being scanned for script porting.

        ![](../figures/migration_process_info.png)

    - After the porting is complete, check the resultant script and porting report.

      ![](../figures/migration_end_info.png)
