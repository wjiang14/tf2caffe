

class Node(object):

    def __init__(self, name, kind, layer=None):
        self.name = name
        self.kind = kind
        self.layer = None
        self.parents = []
        self.children = []
        self.data = None
        self.output_shape = None
        self.metadata = {}

    def add_parent(self, parent_node):
        raise NotImplementedError

    def add_child(self, child_node):
        raise NotImplementedError

    def get_only_parent(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError


class Graph(object):

    def __init__(self, nodes=None, name=None):
        self.nodes = [] if nodes is None else nodes
        self.node_lut = {}
        self.name = Node if name is None else name
        self.sorted_nodes = []

    def add_node(self, node):
        raise NotImplementedError

    def get_node(self, name):
        raise NotImplementedError

    def get_input_nodes(self):
        raise NotImplementedError

    def get_output_nodes(self):
        raise NotImplementedError

    def topologically_sorted(self):
        raise NotImplementedError

    def compute_output_shapes(self):
        raise NotImplementedError

    def replaced(self, new_nodes):
        raise NotImplementedError

    def transformed(self, transformers):
        raise NotImplementedError


class GraphBuilder(object):
    '''Constructs a model graph from a Caffe protocol buffer definition.'''

    def __init__(self, def_path):
        '''
        def_path: Path to the model definition (graph structure)
        data_path: Path to the model data (.caffemodel or ckpt)
        phase: Either 'test' or 'train'. Used for filtering phase-specific nodes.
        '''
        self.def_path = def_path
        #self.load()

    def load(self):
        '''Load the layer definitions from the prototxt.'''
        raise NotImplementedError

    def filter_layers(self, layers):
        '''Filter out layers based on the current phase.'''
        raise NotImplementedError

    def make_node(self, layer):
        '''Create a graph node for the given layer.'''
        raise NotImplementedError

    def make_input_nodes(self):
        '''
        Create data input nodes.

        This method is for old-style inputs, where the input specification
        was not treated as a first-class layer in the prototext.
        Newer models use the "Input layer" type.
        '''
        raise NotImplementedError

    def build(self):
        '''
        Builds the graph from the Caffe layer definitions.
        '''
        # Get the layers
        raise NotImplementedError