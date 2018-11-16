from base.build_graph import Node, Graph, GraphBuilder
from base.shapes import TensorShape, KernelShape
from base.layers import LayerType, NodeKind
from Tensorflow.TFResorver import TFResolver
from tensorflow.core.framework import attr_value_pb2
from queue import Queue
import logging, operator

skip_prefix = [
        "^",
        "train_op",
        "save",
        "gradients",
        "global_step",
        "distort_image",
        "Adagrad",
    ]

skip_scope = [
    "random_uniform",
    "Initializer",
    "optimizer",
    "weight_loss",
    "parallel_read",
    "case"
]

skip_type = [
    "L2Loss",
    "VariableV2",
    "Const",
    "Assign",
    "RandomUniform",
    "FIFOQueueV2"
]


class TFNode(Node):
    def __init__(self, Node_def):
        self.kind = Node_def.op
        self.name = Node_def.name
        self.transform_name = Node_def.name.replace("/", "_")
        self.layer = Node_def
        self.parents = Node_def.input
        self.children = []
        self.if_convert = False
        self.kwargs = {}  # key: value in layer.attr
        self.output_shape = None

    def set_convert_flag(self):
        self.if_convert = True

    @property
    def get_convert_flag(self):
        return self.if_convert

    @property
    def get_scope(self):
        for idx in range(len(self.transform_name)-1, -1):
            if self.transform_name[idx] == "_":
                return self.transform_name[:idx]
        return self.transform_name

    def add_parent(self, parent_node):
        assert parent_node not in self.parents
        self.parents.append(parent_node)
        if self not in parent_node.children:
            parent_node.children.append(self)

    def add_child(self, child_node):
        assert child_node not in self.children
        self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)

    def get_only_parent(self):
        if len(self.parents) != 1:
            raise Exception('Node (%s) expected to have 1 parent. Found %s.' %
                             (self, len(self.parents)))
        return self.parents[0]

    def get_one_parent(self, idx):
        return self.parents[idx]

    @property
    def parameters(self):
        if self.layer is not None:
            return self.layer.attr
        return None

    def get_attr(self, attr_name, default_value = None):
        """
        Warrning: this function only return key: value in node.layer.attr
        :param attr_name: key in node.layer.attr
        :param default_value:
        :return: value in node.layer.layer
        """
        parameters = self.layer.attr
        if attr_name in parameters:
            attr = parameters[attr_name]
            # get value of attr[key]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if isinstance(val, attr_value_pb2.AttrValue.ListValue):
                return list(val.ListFields()[0][1])
            else:
                return val.decode('utf-8') if isinstance(val, bytes) else val
        else:
            return default_value

    def __str__(self):
        return '[%s] %s' % (self.kind, self.name)

    def __repr__(self):
        return '%s (0x%x)' % (self.name, id(self))


class TFGraph(Graph, object):
    def __init__(self, tf_graph_def):
        super(TFGraph, self).__init__()
        self.name = None
        self.model = tf_graph_def
        self.nodes = [TFNode(node) for node in self.model.node]
        self.node_lut = {node.name: node for node in self.nodes}
        self.removed_nodes = []
        self.graph_params = {}

    def add_node(self, node):
        self.nodes.append(node)
        self.node_lut[node.name] = node

    def remove_node(self, node):
        if node in self.nodes:
            print("remove %s: " %node.name)
            self.nodes.remove(node)
            del self.node_lut[node.name]
        else:
            pass
            # logging.warning("Cannot find %s in TFGraph.nodes.", node.name)

    def get_node(self, name):
        try:
            return self.node_lut[name]
        except KeyError:
            raise Exception('Layer (Node) not found: %s' % name)

    def get_input_nodes(self):
        return [node for node in self.nodes if len(node.parents) == 0]

    def get_output_nodes(self):
        return [node for node in self.nodes if len(node.children) == 0]

    def topologically_sorted(self):
        sorted_nodes = []
        q = Queue()
        for node in self.nodes:
            if len(node.parents) == 0 and len(node.children) > 0:
                q.put(node)

        while not q.empty():
            top_node = q.get()
            sorted_nodes.append(top_node)
            for children_name in top_node.children:
                children_node = self.get_node(children_name)
                q.put(children_node)
        return sorted_nodes

    def replaced(self, new_nodes):
        return Graph(nodes=new_nodes, name=self.name)

    def transformed(self, transformers):
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
            if graph is None:
                raise Exception('Transformer failed: {}'.format(transformer))
            assert isinstance(graph, Graph)
        return graph

    def __contains__(self, key):
        return key in self.node_lut


class TFGraphBuilder(GraphBuilder, object):
    '''Constructs a model graph from a Caffe protocol buffer definition.'''

    def __init__(self, *args, data_path, def_path=None, py_net_def=None, net_name=None):
        '''
        def_path: Path to the model definition (.meta)
        data_path: Path to the model data (.ckpt)
        '''
        self.def_path = def_path
        self.data_path = data_path
        self.py_net_def = py_net_def
        self.net_spec = args
        self.net_name = net_name
        self.TFGraph = None
        self.params = None
        self.load()

    def load(self):
        if self.data_path is not None and self.def_path is not None:
            graph = TFResolver(self.data_path, self.def_path)
            self.TFGraph, self.params = graph.load()
        elif self.data_path is not None and self.py_net_def is not None and self.net_name is not None:
            graph = TFResolver(self.data_path)
            graph.name = self.net_name
            batch_size = self.net_spec[0][0]
            input_width = self.net_spec[0][1]
            input_height = self.net_spec[0][2]
            input_channel = self.net_spec[0][3]
            self.TFGraph, self.params = graph.load([self.py_net_def, self.net_name, batch_size, input_width, input_height, input_channel])
        else:
            raise Exception("Only two modes are supported, .ckpt and .meta files or .ckpt with .py Net definition file."
                            "Check your model initialization definition.")

    def filter_layers(self, src_node):
        '''Filter out layers based on the current phase.'''
        filtered_layers = []
        for prefix in skip_prefix:
            if src_node.name.startswith(prefix):
                return True

    def make_node(self, layer):
        '''Create a graph node for the given layer.'''
        raise NotImplementedError

    def get_node_scope(self, node_name):
        return node_name.split("/")

    def make_input_nodes(self):
        '''
        Create data input nodes.

        This method is for old-style inputs, where the input specification
        was not treated as a first-class layer in the prototext.
        Newer models use the "Input layer" type.
        '''
        raise NotImplementedError

    def convert_Placeholder(self, node):
        node.set_convert_flag()
        layer_shape = node.layer.attr['shape'].shape.dim
        tensor_shape = len(layer_shape)
        temp = []
        for idx in range(0, tensor_shape):
            temp.append(int(layer_shape[idx].size))
        node.output_shape = TensorShape._make(temp)
        logging.info("Finished converting %s", node.name)

    def convert_Conv2D(self, node):
        node.set_convert_flag()
        node.kwargs["value_type"] = node.layer.attr["T"].type
        node.kwargs["data_format"] = node.get_attr("data_format")
        node.kwargs["dilations"] = node.get_attr("dilations")  # list: default-> [1, 1, 1, 1]
        node.kwargs["padding"] = node.get_attr("padding")  # string: default-> "SAME" or "VALID"
        node.kwargs["strides"] = node.get_attr("strides")  # list:
        node.kwargs["if_bias"] = True if node.children[0][-7:] == "BiasAdd" else False

        if not operator.eq(node.kwargs["dilations"], [1, 1, 1, 1]):
            raise Exception("Current do not support dilated convolution.")

        if len(node.parents) >= 2 and node.parents[1][-4:] == "read":
            conv_node = self.tf_graph.get_node(node.parents[1][:-5])
            kernal_shape = conv_node.get_attr("shape")
            node.kwargs["kernal_shape"] = KernelShape._make([kernal_dim.size for kernal_dim in kernal_shape.dim])
        else:
            raise Exception("Input of Conv2D should be weights/read, check your weights input(%s) of Conv2D layer[op]" %node.parents[1])
        logging.info("Finished converting %s", node.name)

    def convert_RealDiv(self, node):
        # deal with dropout layer
        # In tensorflow, the dropout layer is consist of several ops, such as Mul, Add, Floor, RealDiv, et al.
        # To implement dropout, tensorflow need a keep_prob to generate a random mask, we need scope information to
        # deal with dropout layer, I cannot find a better solution currently.
        if "dropout" in node.name:
            node.set_convert_flag()
            keep_prob_node_name = node.parents[1]
            keep_prob_node = self.tf_graph.get_node(keep_prob_node_name)
            assert keep_prob_node.layer.op == "Const"
            node.kwargs['keep_prob'] = float(keep_prob_node.layer.attr['value'].tensor.float_val[0])
        logging.info("Finished converting %s", node.name)

    def convert_Relu(self, node):
        node.set_convert_flag()
        logging.info("Finished converting %s", node.name)

    def convert_Softmax(self, node):
        node.set_convert_flag()
        logging.info("Finished converting %s", node.name)

    def convert_MaxPool(self, node):
        node.set_convert_flag()
        logging.info("Finished converting %s", node.name)

    def convert_AvgPool(self, node):
        node.set_convert_flag()
        logging.info("Finished converting %s", node.name)

    def convert_BiasAdd(self, node):
        node.set_convert_flag()
        logging.info("Finished converting %s", node.name)

    def convert_Mul(self, node):
        node.set_convert_flag()
        scopes = node.name.split("/")[:-1]
        if "dropout" in scopes:
            pass
        else:
            # TODO need to write Mul op code
            pass

    def convert_Squeeze(self, node):
        node.set_convert_flag()
        logging.info("Finished converting %s", node.name)

    def build(self):
        '''
        Builds the graph from the Caffe layer definitions.
        '''
        # Get the layers
        # if Graph_def is successfully loaded, then skip unused graph node (layer), build graph
        src_tf_graph = self.TFGraph.as_graph_def()
        self.tf_graph = TFGraph(src_tf_graph)

        # # skip unnecessary node in graph.nodes
        for node in self.tf_graph.nodes:
            # connect nodes in graph
            for parent_node_name in node.parents:
                if parent_node_name not in self.tf_graph.node_lut:
                    logging.warning("%s not in node_lut", parent_node_name)
                else:
                    parent_node = self.tf_graph.get_node(parent_node_name)
                    assert parent_node_name != node.name
                    parent_node.children.append(node.name)

        exclude_nodes = []
        for node in self.tf_graph.nodes:
            NodeKind.compute_output_shape(node, self.tf_graph)
            src_node_name = node.name
            for skip_node_prefix in skip_prefix:
                if src_node_name.startswith(skip_node_prefix):
                    # tf_graph.remove_node(node)
                    exclude_nodes.append(src_node_name)
                    continue

            for skip_node_scope in skip_scope:
                tf_variable_scope = self.get_node_scope(src_node_name)
                if skip_node_scope in tf_variable_scope:
                    # tf_graph.remove_node(node)
                    exclude_nodes.append(src_node_name)
                    continue

            if src_node_name not in exclude_nodes:
                self.tf_graph.removed_nodes.append(node)

            # convert kind of layers;
            if hasattr(self, "convert_" + node.kind):
                func = getattr(self, "convert_" + node.kind)
                func(node)
            else:
                continue

