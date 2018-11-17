from Tensorflow.TFGraph import TFGraphBuilder
from six import string_types as _string_types
import numpy as np


class EmitOps():
    def __init__(self, graph, tab=None):
        # tab should be \t or "    "
        self.TFGraph = graph
        self.tab = tab if tab is '\t' else ' ' * 4

    def _get_pad(self, node):
        assert node.kind == "Conv2D" or node.kind in ["MaxPool", "AvgPool"]
        pad_h = 0
        pad_w = 0

        if node.kwargs["padding"] == "SAME":
            input_node_name = node.parents[0]
            input_node = self.TFGraph.tf_graph.get_node(input_node_name)
            input_shape = input_node.output_shape
            out_height = np.ceil(float(input_shape.height) / float(node.kwargs["strides"][1]))
            out_width = np.ceil(float(input_shape.width) / float(node.kwargs["strides"][2]))

            pad_along_height = max((out_height - 1) * node.kwargs["strides"][1] +
                                   node.kwargs['kernal_shape'].kernel_h - input_shape.height, 0)
            pad_along_width = max((out_width - 1) * node.kwargs["strides"][1] +
                                  node.kwargs['kernal_shape'].kernel_w - input_shape.width, 0)

            # top_pad
            pad_h = int(pad_along_height // 2)
            # pad_bottom = pad_along_height - pad_top
            # left_pad
            pad_w = int(pad_along_width // 2)
            # pad_right = pad_along_width - pad_left
        elif node.kwargs["padding"] == "VALID":
            # Valid do not add extral pad during convolution
            pass
        else:
            raise Exception("Wrong padding option: %s, currently padding option only ['SAME', 'VALID']",
                            node.kwargs["padding"])
        return pad_h, pad_w

    def emit_Placeholder(self, node, test=True):
        data_code = "net.{}, net.{} = L.Data(source = {}, backend = caffe.params.Data.LMDB, batch_size = {}, ntop = 2, \
transform_param = dict(crop_size = {}, mirror = True))".format(
            "data",
            "label",
            "input_file",
            "batch_size",
            node.output_shape.height)
        node.name = 'data'
        # children_node_name = node.children[0]
        # children_node = self.TFGraph.tf_graph.get_node(children_node_name)
        # print(children_node.name, children_node.kind, children_node.parents)
        # children_node.parents[0] = 'data'
        return data_code

    def emit_Conv2D(self, node):
        pad_h, pad_w = self._get_pad(node)

        if "fc" in node.get_scope:
            data_code = "net.{:<15} = L.InnerProduct(net.{}, num_output={}, bias_term={}, ntop=1)".format(
                node.transform_name,
                node.parents[0].replace('/', '_'),
                node.output_shape.channel,
                node.kwargs["if_bias"])
        else:
            data_code = "net.{:<15} = L.Convolution(net.{}, kernel_h={}, kernel_w={}, stride={}, num_output={}, pad_h={}, pad_w={}, group={}, \
bias_term={}, ntop=1)".format(
                node.transform_name,
                node.parents[0].replace('/', '_'),
                node.kwargs["kernal_shape"].kernel_h,
                node.kwargs["kernal_shape"].kernel_w,
                node.kwargs["strides"][1],
                node.output_shape.channel,
                pad_h,
                pad_w,
                0,
                node.kwargs["if_bias"])
        return data_code

    def emit_Relu(self, node):
        parent_node_name = node.parents[0]
        for idx in range(len(parent_node_name) - 1, -1, -1):
            if parent_node_name[idx] == "/":
                last_term = parent_node_name[idx+1:]
                if last_term == "BiasAdd":
                    # there is BiasAdd op before ReLU, then we go to grad father node
                    parent_node = self.TFGraph.tf_graph.get_node(parent_node_name)
                    parent_node_name = parent_node.parents[0]
                    break
        data_code = "net.{:<15} = L.ReLU(net.{}, in_place={}, ntop=1)".format(
            node.transform_name,
            parent_node_name.replace('/', '_'),
            "True")
        return data_code

    def emit_MaxPool(self, node):
        pooling_type = node.kind
        if pooling_type == 'MaxPool':
            pooling_type = 'P.Pooling.MAX'
        elif pooling_type == 'AvgPool':
            pooling_type = 'P.Pooling.AVE'
        else:
            raise ValueError()

        input_node_name = node.parents[0]
        input_node = self.TFGraph.tf_graph.get_node(input_node_name)
        input_shape = input_node.output_shape
        if_global_pooling = (input_shape.height == node.kwargs['kernel_shape'].kernel_h and
                             input_shape.width == node.kwargs['kernel_shape'].kernel_w)

        pad_h, pad_w = self._get_pad(node)
        if if_global_pooling:
            data_code = "net.{:<15} = L.Pooling(net.{}, pool={}, stride={}, global_pooling=True, ntop=1)".format(
                node.transform_name,
                node.parents[0].replace('/', '_'),
                pooling_type,
                node.kwargs['strides'][1])
        else:
            data_code = "net.{:<15} = L.Pooling(net.{}, pool={}, kernel_size={}, pad_h={}, pad_w={}, stride={}, ntop=1)".format(
                              node.transform_name,
                              node.parents[0].replace('/', '_'),
                              pooling_type,
                              node.kwargs['kernel_shape'].kernel_h,
                              pad_h,
                              pad_w,
                              node.kwargs['strides'][1])
        return data_code

    def emit_RealDiv(self, node):
        parent_node_name = node.parents[0].replace('/', '_')
        keep_prob = node.kwargs['keep_prob']
        # we only consider dropout here, tensorflow do not have dropout op, but consist of several ops
        if "dropout" in node.name:
            while node.if_convert and node.kind in ['RealDiv', 'Mul']:
                child_node_name = node.children[0]
                node = self.TFGraph.tf_graph.get_node(child_node_name)
            data_code = "net.{:<15} = L.Dropout(net.{}, dropout_ratio={}, in_place=True)".format(
                node.parents[0].replace('/', '_'),
                parent_node_name,
                keep_prob
            )
        return data_code


class TFEmitter(EmitOps):
    def __init__(self, graph, tab=None):
        super(TFEmitter, self).__init__(graph, tab)
        self.phase = None
        self.body_code = ""

    def indent(self):
        self.body_code += self.tab

    def emit_imports(self):
        self.body_code += """import caffe
from caffe import layers as L
from caffe import params as P

"""

    def emit_generate_proto(self):
        self.body_code += """def create_neural_net(input_file, batch_size):\n"""
        self.indent()
        self.body_code += """net = caffe.NetSpec()\n"""

    def write_code(self, saved_code_path):
        with open(saved_code_path, "w") as f:
            f.write(self.body_code)
        print("Save caffe code to %s" %saved_code_path)

    def save_weights(self, save_code_path):
        with open(save_code_path, 'wb') as f:
            np.save(f, self.TFGraph.weights)
        print("Save weights data to %s" %save_path)

    def run(self):
        pass

    def add_body(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = [codes]
        for code in codes:
            self.body_code += (self.tab * indent) + code + '\n'

    def gen_code(self, phase):
        self.phase = phase
        self.emit_imports()
        # generate prototxt file
        self.emit_generate_proto()

        for node in self.TFGraph.tf_graph.removed_nodes:
            # node = self.TFGraph.tf_graph.get_node(layer_name)
            if node.if_convert:
                if hasattr(self, "emit_" + node.kind):
                    func = getattr(self, "emit_" + node.kind)
                    code = func(node)
                    self.add_body(1, code)
                else:
                    print("Currently do not support %s emmiter" %node.kind)
        print(self.body_code)


if __name__ == "__main__":
    ckpt = "../pretrain_model/vgg_16.ckpt"
    py_file = "nets.vgg"
    save_path = "../converted_model/caffe_code.py"
    net_spect = [1, 224, 224, 3]
    graph = TFGraphBuilder(net_spect, data_path=ckpt, py_net_def=py_file, net_name='vgg_16')
    graph.build()
    # params = graph.params
    # for layer in params.get_variable_to_shape_map():
    #     print(layer)
    emmiter = TFEmitter(graph=graph)
    emmiter.save_weights("../converted_model/weights.npy")
    emmiter.gen_code(phase="Test")
    emmiter.write_code(save_path)