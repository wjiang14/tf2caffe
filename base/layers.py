import re
import numbers
from collections import namedtuple


class NodeKind(object):

    @staticmethod
    def map_raw_kind(kind):
        raise NotImplementedError

    @staticmethod
    def compute_output_shape(node):
        raise NotImplementedError


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
