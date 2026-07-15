# Reading Node Names from a PB Model File

You can use a simple script to obtain the node names defined in a .pb model file. Specifically, create the  **readNodeName_from_pb.py**  script, write the following code to the script, and run the script.

```python
import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = 'resnet50_tensorflow_1.5.pb'        # Set the directory of the .pb file as needed.
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        for i, node in enumerate(graph_def.node):
            print("Name of the node : %s" % node.name)
```

Run the following command:

`python3.7 readNodeName_from_pb.py`

The node names in the model file can be output.

This script can only be used to view node names in a model. You can use a model visualization tool \(for example,  [Netron](https://netron.app)\) to view the network topology.
