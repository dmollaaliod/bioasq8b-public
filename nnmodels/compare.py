"Comparison between sentence and question"
import tensorflow as tf

class SimMul:
    """Implementation of a simple similarity based on multiplication"""
    size = None
    name = "SimMul"
    def compare(self, sentence, question, size):
        self.size = size
        return tf.multiply(sentence, question)

class SimEuc:
    """Implementation of a simple similarity based on Euclidean distance"""
    size = None
    name = "SimEuc"
    def compare(self, sentence, question, size):
        self.size = size
        return (sentence - question) ** 2

class SimYu:
    """Implementation of a similarity based on Yu's paper"""
    size = 1
    name = "SimYu"

    def compare(self, sentence, question, size):
        with tf.name_scope('sim_yu'):
            W_sim = tf.Variable(tf.truncated_normal([size, size],
                                                stddev = 0.1))
            b_sim = tf.Variable(tf.constant(0.1, shape=[1]))

            return tf.reduce_sum(tf.multiply(tf.matmul(question, W_sim),
                                             sentence),
                                 1, keep_dims=True) + b_sim
