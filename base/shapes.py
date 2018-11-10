from collections import namedtuple
import numpy as np
import math, logging
TensorShape = namedtuple('TensorShape', ['batch', 'height', 'width', 'channel'])
KernelShape = namedtuple('KernelShape', ['kernel_h', 'kernel_w', 'in_channel', 'out_channel'])


def get_strided_kernel_output_shape(node, graph):
    if len(node.parents) >= 2 and node.parents[1][-4:] == "read":
        conv_node = graph.get_node(node.parents[1][:-5])
        kernal_shape = conv_node.get_attr("shape")
        kernal_shape = KernelShape._make([kernal_dim.size for kernal_dim in kernal_shape.dim])
    else:
        raise Exception(
            "Input of Conv2D should be weights/read, check your weights input(%s) of Conv2D layer[op]" % node.parents[1])

    # calculate ouput_shape of Conv2D layer[op] after Convolution
    # reference: https://www.tensorflow.org/api_guides/python/nn#Convolution
    input_node_name = node.layer.input[0]
    input_node = graph.get_node(input_node_name)
    if node.get_attr("padding") == "VALID":
        out_height = np.ceil(float(input_node.output_shape.height - kernal_shape.kernel_h + 1)
                             / float(node.get_attr("strides")[1]))
        out_width = np.ceil(float(input_node.output_shape.width - kernal_shape.kernel_w + 1)
                            / float(node.get_attr("strides")[2]))
    else:
        if node.get_attr("padding") != "SAME":
            print("Currently do not support %s, convert Conv2D as %s" % (node.get_attr("padding"), "SAME"))
        out_height = np.ceil(float(input_node.output_shape.height) / float(node.get_attr("strides")[1]))
        out_width = np.ceil(float(input_node.output_shape.width) / float(node.get_attr("strides")[2]))
    node.output_shape = TensorShape(input_node.output_shape.batch, out_height, out_width, kernal_shape.out_channel) \
        if node.get_attr("data_format") == "NHWC" else TensorShape(input_node.output_shape.batch, kernal_shape.out_channel, out_height, out_width)


def convert_pooling(node, graph, pool_type):
    node.kwargs["value_type"] = node.layer.attr["T"].type
    node.kwargs["data_format"] = node.get_attr("data_format")
    # padding
    node.kwargs["padding"] = node.get_attr("padding")  # string: default-> "SAME" or "VOID"
    # strides
    node.kwargs['strides'] = node.get_attr('strides')
    # window_shape
    # KernelShape tuple is [h, w, i_channel, o_channel];
    # In pooling layer, we want to do pooling for h and w, which means ksize at pooling is [1, h, w, 1]
    node.kwargs['kernel_shape'] = KernelShape(node.get_attr('ksize')[1], node.get_attr('ksize')[2], node.get_attr('ksize')[0], node.get_attr('ksize')[3])
    # pool type
    node.kwargs['pooling_type'] = pool_type

    # calculate output_shape of after Pooling layer[op]
    input_node_name = node.layer.input[0]
    input_node = graph.get_node(input_node_name)
    if node.get_attr("padding") == "VALID":
        out_height = np.ceil(float(input_node.output_shape.height - node.kwargs["kernel_shape"].kernel_h + 1)
                             / float(node.get_attr("strides")[1]))
        out_width = np.ceil(float(input_node.output_shape.width - node.kwargs["kernel_shape"].kernel_w + 1)
                            / float(node.get_attr("strides")[2]))
    else:
        if node.get_attr("padding") != "SAME":
            logging.warning("Currently do not support %s, convert Conv2D as %s" %(node.get_attr("padding"), "SAME"))
        out_height = np.ceil(float(input_node.output_shape.height) / float(node.get_attr("strides")[1]))
        out_width = np.ceil(float(input_node.output_shape.width) / float(node.get_attr("strides")[2]))

    node.output_shape = TensorShape(input_node.output_shape.batch, out_height, out_width, node.kwargs["kernel_shape"].out_channel) \
        if node.get_attr("data_format") == "NHWC" else TensorShape(input_node.output_shape.batch, node.kwargs["kernel_shape"].out_channel,
                                                       out_height, out_width)


def shape_not_implemented(node, graph=None):
    raise NotImplementedError


def shape_identity(node, graph=None):
    if len(node.parents) == 0:
        shape_scalar(node)
    else:
        input_node_name = node.parents[0]
        input_node = graph.get_node(input_node_name)
        node.output_shape = input_node.output_shape


def shape_scalar(node, graph=None):
    node.output_shape = TensorShape(1, 1, 1, 1)


def shape_convolution(node, graph=None):
    get_strided_kernel_output_shape(node, graph)


def shape_maxpool(node, graph=None):
    convert_pooling(node, graph, b'MAX')


def convert_avgpool(node, graph=None):
    convert_pooling(node, graph, b'AVG')


def shape_squeeze(node, graph=None):
    squeeze_dims = node.get_attr("squeeze_dims")
    node.kwargs["squeeze_axis"] = squeeze_dims
    input_node_name = node.layer.input[0]
    input_node = graph.get_node(input_node_name)
    if len(squeeze_dims) == 2:
        temp = []
        for idx, shape in enumerate(input_node.output_shape):
            if idx >= squeeze_dims[0] and idx <= squeeze_dims[1] and shape == 1:
                temp.append(0)
            else:
                temp.append(int(shape))
        node.output_shape = TensorShape._make(temp)
    else:
        node.output_shape = TensorShape._make([shape for shape in input_node.output_shape if shape != 1])


def shape_placeholder(node, graph=None):
    assert len(node.parents) == 0
    layer_shape = node.layer.attr['shape'].shape.dim
    tensor_shape = len(layer_shape)
    temp = []
    for idx in range(0, tensor_shape):
        temp.append(int(layer_shape[idx].size))
    node.output_shape = TensorShape._make(temp)