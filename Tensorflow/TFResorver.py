import os, logging
from tensorflow.python import pywrap_tensorflow
import importlib

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)


class TFResolver(object):
    def __init__(self, tf_graph_data=None, tf_graph_def=None):
        self.tf_graph_def = tf_graph_def  # .meta is path of Tensorflow meta graph
        self.tf_graph_data = tf_graph_data  # .ckpt is path of Tensorflow ckpt checkpoint

    def load(self, *args):
        """
        :param args: 0: path of py file (TF net defination); 1: function name, 2: batch_size, 3: height, 4: width, 5: channel
        :return:
        """
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise Exception("Tensorflow import error, please install it before convert tf model.")

        with tf.Session() as sess:
            self.tf_reader = pywrap_tensorflow.NewCheckpointReader(self.tf_graph_data)
            if self.tf_graph_def is not None and self.tf_graph_data is not None:
                assert os.path.splitext(self.tf_graph_data)[1] == ".ckpt" and os.path.splitext(self.tf_graph_def)[1] == ".meta"
                saver = tf.train.import_meta_graph(self.tf_graph_def)
                saver.restore(sess, self.tf_graph_data)
                logging.info("Load Tensorflow checkpoint %s and %s" % self.tf_graph_def and self.tf_graph_data)

            elif os.path.splitext(self.tf_graph_data)[1] == ".ckpt" and self.tf_graph_def is None:
                try:
                   lib = importlib.import_module(args[0][0])  # path of Tensorflow net defination .py file.
                except ImportError:
                    raise Exception("Fail to import net py file, check your py path or use .meta instead.")

                tf_input = tf.placeholder(dtype=tf.float32, shape=args[0][2:6])
                if hasattr(lib, args[0][1]):
                    fun = getattr(lib, args[0][1])
                    fun(tf_input)
                saver = self.tf.train.Saver()
                saver.restore(sess, self.tf_graph_data)
                logging.info("Load Tensorflow checkpoint %s only" % self.tf_graph_data)
            else:
                raise Exception("To load Tensorflow model, .py or .meta file must be provided")
            self.graph = tf.get_default_graph()
            logging.info("Tensorflow graph %s is loaded" %self.graph)
        return self.graph, self.tf_reader
