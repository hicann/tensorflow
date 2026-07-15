# Script Implementation Process and Key Files

## Training Script Implementation Process

The ResNet-50 original network script is an API of the Estimator model and is a high-level API of TensorFlow. The implementation process of this training script is as follows:

| No. | Step | Description |
| --- | --- | --- |
| 1 | Preprocess data. | Create the input function input_fn. |
| 2 | Construct a model. | Construct the model function model_fn. |
| 3 | Set run configuration. | Instantiate Estimator and pass an object of the Runconfig class as the run parameter. |
| 4 | Start training. | Call the training method Estimator.train() in Estimator to train the model with a fixed number of steps using the specified input. |

## Key File Overview

The key file directory is organized as follows. \(Only some files that need to be modified are listed. For additional files, see the original ResNet script.\)

```text
├── r1
│   ├── resnet       // ResNet main directory.
│        ├── imagenet_main.py      // Script for training the network based on the ImageNet dataset.
│        ├── imagenet_preprocessing.py     // ImageNet preprocessing module.
│        ├── resnet_model.py     // ResNet model file.
│        ├── resnet_run_loop.py    // Data input processing and execution iteration (training, validation, and test).
├── utils
│   ├── flags
│   │   ├── _base.py     // Defines the common parameters and sets the default value.
```

| File name | Description |
| --- | --- |
| imagenet_main.py | Contains APIs related to ImageNet preprocessing, model construction definition, and model runtime. The get_filenames(), parse_record(), input_fn(), get_synth_input_fn(), and _parse_example_proto() functions are used for data preprocessing. The ImagenetModel class, imagenet_model_fn(), run_cifar(), and define_cifar_flags() functions are used for model operations. |
| imagenet_preprocessing.py | Contains ImageNet image data preprocessing APIs for sampling training images with the provided bounding box, cropping images based on the bounding box, randomly flipping images, and adjusting images to the target output size (the aspect ratio is not retained). Image resizing (aspect ratio retained) and central cropping are used during the evaluation process. |
| resnet_model.py | Implements the ResNet model, including the auxiliary functions for ResNet model construction and ResNet block definition functions. |
| resnet_run_loop.py | Model runtime file, including input processing and execution iteration. Input processing includes decoding the input data, converting the format, outputting images and labels, as well as setting data randomization, batch, and pre-reading based on whether it is the training scenario. Execution iteration includes constructing Estimator, and performing training and validation. In general, the model is run in specific environments to implement data and error flowing, so that model parameters can be updated using gradient descent. |
