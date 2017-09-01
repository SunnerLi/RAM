from config import *
import tensorlayer as tl
import tensorflow as tf

weight = []
bias = []

class Dense(object):
    def __new__(self, input_tensor, n_units=2, activation_fn=tf.identity, name=None):
        global weight
        global bias
        if name == None:
            print('error! You should assign name of variable scope')
            exit()
        with tf.variable_scope(name) as vs:
            W = tf.Variable(tf.random_normal((int(input_tensor.get_shape()[1]), n_units), mean=0.0), dtype=tf.float32)
            b = tf.Variable(tf.random_normal((n_units,), dtype=tf.float32))            
            weight.append(W)
            bias.append(b)
            return tf.add(tf.matmul(input_tensor, W), b)

if __name__ == '__main__':
    _input = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
    layer = Dense(_input, n_units=16)