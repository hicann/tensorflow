# Developing a Custom Operator

If the TensorFlow network contains operators that are not supported by CANN, you can customize Ascend C operators and adapt them to the TensorFlow framework. The following figure shows the process.

![](../figures/customop.png)

1. Develop a custom operator.

    Develop a custom operator based on Ascend C. The development process includes the following steps:

    1. Create an operator project.
    2. Implement the operator, including operator prototype definition, operator implementation on the kernel, and tiling implementation on the host.
    3. Integrate the operator into a graph by implementing the corresponding adaptation functions such as shape inference.

2. Develop the TensorFlow adaptation plugin.

    Register the custom operator by calling  **REGISTER_CUSTOM_OP**  and map the TensorFlow custom operator to the CANN operator. In this case, you also need to develop the TensorFlow custom operator.

3. Build and deploy the operator project. Build and generate the custom operator installation package, install the operator package, and deploy the custom operator to the operator acceleration library.
4. Call the operator in TensorFlow.

For details about the preceding process, see [Ascend C Operator Development](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).
