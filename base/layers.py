import re
import numbers
from collections import namedtuple
from base.shapes import *

LAYER_DESCRIPTORS = {

    # Tensorflow op Types
    'MaxPool': shape_maxpool,
    'VariableV2': shape_identity,
    'Mul': shape_identity,
    'RestoreV2': shape_identity,
    'Relu': shape_identity,
    'Sub': shape_identity,
    'Identity': shape_identity,
    'Const': shape_identity,
    'Assign': shape_identity,
    'Squeeze': shape_squeeze,
    'Placeholder': shape_placeholder,
    'Fill': shape_identity,
    'Conv2D': shape_convolution,
    'RealDiv': shape_identity,
    'RandomUniform': shape_identity,
    'Add': shape_identity,
    'Floor': shape_identity,
    'SaveV2': shape_identity,
    'BiasAdd': shape_identity,
    'NoOp': shape_scalar,
}

LAYER_TYPES = LAYER_DESCRIPTORS.keys()

LayerType = type('LayerType', (), {t: t for t in LAYER_TYPES})


class NodeKind(LayerType):
    @staticmethod
    def map_raw_kind(kind):
        if kind in LAYER_TYPES:
            return kind
        return None

    @staticmethod
    def compute_output_shape(node, graph):
        try:
            LAYER_DESCRIPTORS[node.kind](node, graph)
        except NotImplementedError:
            raise Exception('Output shape computation not implemented for type: %s' % node.kind)


class NodeDispatchError(Exception):

    pass


class NodeDispatch(object):

    @staticmethod
    def get_handler_name(node_kind):
        raise NotImplementedError

    def get_handler(self, node_kind, prefix):
        raise NotImplementedError


class LayerAdapter(object):

    def __init__(self, layer, kind):
        self.layer = layer
        self.kind = kind

    @property
    def parameters(self):
        raise NotImplementedError

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        raise NotImplementedError

    @property
    def kernel_parameters(self):
        raise NotImplementedError


KernelParameters = namedtuple('KernelParameters', ['kernel_h', 'kernel_w', 'stride_h', 'stride_w',
                                                   'pad_h', 'pad_w'])
