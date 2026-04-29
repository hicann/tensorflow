# 读取pb模型文件的节点名称

当用户无法获取.pb模型文件的节点名称时，可创建readNodeName_from_pb.py脚本并写入如下代码并执行。

```python
import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = 'resnet50_tensorflow_1.5.pb'        # .pb模型文件路径，用户可根据实际修改
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        for i, node in enumerate(graph_def.node):
            print("Name of the node : %s" % node.name)
```

执行如下命令：

**python3.7 readNodeName_from_pb.py**

则可以输出模型文件中的节点名称。

此脚本仅支持查看模型中的节点名称，您可以通过模型可视化工具（例如：[Link](https://netron.app)）查看网络的拓扑结构。
