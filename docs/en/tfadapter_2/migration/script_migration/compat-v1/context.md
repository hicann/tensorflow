# Context

TensorFlow 2.6.5 provides the  **compat.v1**  module to enable TensorFlow 1._x_  API compatibility. In other words, you can use TensorFlow 1._x_  APIs in TensorFlow 2.6.5.

There may be TensorFlow 2.6.5 scripts that call TensorFlow 1._x_  APIs using  **tf.compat.v1**  to control script execution in session mode \(Estimator/Session/Keras\) used in TensorFlow 1._x_. Such training scripts can be ported to  AI processor, and the porting points are basically the same as those of TF1.
