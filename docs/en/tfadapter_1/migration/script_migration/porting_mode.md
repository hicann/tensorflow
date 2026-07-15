# Porting Modes

You have the following two options for porting your training script developed based on the TensorFlow Python API to the  AI processor:

- [Automated porting](automated_porting.md)

    Algorithm engineers can use the porting tool to analyze the support for TensorFlow and Horovod Python APIs on the  AI processor, and automatically port native TensorFlow training scripts to those supported by the  AI processor. For APIs unportable by the tool, modify your training scripts according to the tool report.

- [Manual porting](manual_porting.md)

    Algorithm engineers can modify TensorFlow training scripts to adapt them to the  AI processor. This method is more complex. The more friendly automated porting mode is recommended.
