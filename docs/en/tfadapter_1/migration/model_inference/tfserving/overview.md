# Overview

TensorFlow Serving \(TF Serving\) is a flexible and high-performance service system designed for machine learning models in production environments. It uses models in SavedModel format, provides RESTful APIs and gRPC external APIs, and depends on the TensorFlow source code. The official website is  [https://www.tensorflow.org/tfx/guide/serving](https://www.tensorflow.org/tfx/guide/serving).

TF Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. It provides out-of-the-box integration with TensorFlow models, and can be easily extended to serve other types of models and data. Considering performance, its core code and the dependent TensorFlow are developed using C++.

![](../../figures/tfserving_overview.png)

This section describes how to compile the TF Adapter and TF Serving source code, enabling TF Serving to load the TF Adapter plugin via TensorFlow and use Ascend AI Processors for online inference.
