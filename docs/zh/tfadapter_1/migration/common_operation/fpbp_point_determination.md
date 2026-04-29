# 如何获取fp_point与bp_point

fp_point指的是forward propagation，bp_point指的是backward propagation，此处特指网络计算图的第一个前向传播算子和最后一个反向传播算子。当用户需要采集训练迭代轨迹数据对训练任务进行性能分析时，需要输入前后向计算的打点算子，从而可以获取前后向计算时间。

在训练过程中，通常我们保存的结果是checkpoint和meta文件，但在分析前向和反向算子的时候，需要网络结构文件（graph.pbtxt），一般用tf.io.write_graph来保存计算图，如果用Estimator启动训练的话，在RunConfig中配置model_dir参数也能打印计算图文件。

打开graph.pbtxt文件：

![](../figures/graph_pbtxt.png)

每一个“node”指的是一个算子，我们要观察的是“op”这个属性：

![](../figures/graph_pbtxt_node.png)

获取fp_point时，从第一个node开始搜索，找到第一个“计算类”算子。由于需要获取实际训练的性能数据，因此打点算子需要排除“数据类型”节点和“存储类”节点，诸如Const， VariableV2，IteratorV2，Identity，Reshape，Case等算子，或者说“name”这个属性中带有“step”、“Dataset”“seed”“kernel”的算子都可以排除。例如：

![](../figures/graph_pbtxt_exclude.png)

到1732行时找到第一个计算类算子（op字段为“MatMul”，代表矩阵乘）。“fp_point”值为算子的“name”，也就是“dense/MatMul”。

获取bp_point时，从图的最后一个node开始查找，找到第一个出现的“gradients”的计算图的计算节点，该算子即可作为反向打点算子。同样，排除“op”字段值是Assign、Const、NoOp等。例如：

![](../figures/graph_pbtxt_gradients.png)

找到4457行才找到最后一个反向传播算子，op是MatMul，name中带有gradients，bp_point取“gradients/dense/MatMul_grad/MatMul_1”。

找到的打点算子可能会被融合或改名，因此找到该算子之后，最好和GE生成的图ge_proto_xxxxx_Build.txt做下比较，如果能匹配上，则可以直接填入；如果名字无法匹配（例如GE生成的图多一个_1），则以GE中的算子名称为准。
