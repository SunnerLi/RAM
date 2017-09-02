from config import *
import tensorlayer as tl
import tensorflow as tf

class GlimpsNetwork(object):
    def __init__(self, location_ph):
        self.location_ph = location_ph

    def getGlimps(self, loc_tensor):
        self.glimps_imgs = tf.reshape(self.location_ph, [tf.shape(self.location_ph)[0], 28, 28, 1], name='reshape_layer_1')
        self.glimps_imgs = tf.image.extract_glimpse(self.glimps_imgs, [win_size, win_size], loc_tensor)
        self.glimps_imgs = tf.reshape(self.glimps_imgs, [tf.shape(loc_tensor)[0], win_size * win_size * 1])
        return self.glimps_imgs

    def __call__(self, loc_tensor):
        self.retina_imgs = self.getGlimps(loc_tensor)

        # Glimps network upper part
        self.retina_net = tl.layers.InputLayer(self.retina_imgs, name='retina_input_layer')
        self.retina_net = tl.layers.DenseLayer(self.retina_net, n_units = 128, name ='glimps_ind_fc_1')
        self.retina_net = tl.layers.BatchNormLayer(self.retina_net, name='glimps_ind_bn_1')
        self.retina_net = tf.nn.relu(self.retina_net.outputs, name='glimps_ind_relu_1')

        # Glimps network lower part
        self.location_net = tl.layers.InputLayer(loc_tensor, name='location_part_input_layer')
        self.location_net = tl.layers.DenseLayer(self.location_net, n_units = 128, name ='glimps_ind_fc_2')
        self.location_net = tl.layers.BatchNormLayer(self.location_net, name='glimps_ind_bn_2')
        self.location_net = tf.nn.relu(self.location_net.outputs, name='glimps_ind_relu_2')

        # Glimps network right part
        self.glimps_net = tf.concat([self.retina_imgs, self.location_net], axis=-1)       
        self.glimps_net = tl.layers.InputLayer(self.glimps_net, name='glimps_merge_input_layer')
        self.glimps_net = tl.layers.DenseLayer(self.glimps_net, n_units = 256, act = tf.nn.relu, name='glimps_fc_1')

        return self.glimps_net.outputs

class LocationNetwork(object):
    def __call__(self, state_tensor):
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            # Network structure
            self.location_net = tl.layers.InputLayer(state_tensor)
            self.location_net = tl.layers.DenseLayer(self.location_net, n_units = loc_dim, name='location_net_fc1')

            # Add random
            self.mean = tf.stop_gradient(tf.clip_by_value(self.location_net.outputs, -1.0, 1.0))
            self.location = self.mean + tf.random_normal((tf.shape(state_tensor)[0], loc_dim), stddev=loc_std)
            self.location = tf.stop_gradient(self.location)
            return self.location, self.mean

if __name__ == '__main__':
    location_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = LocationNetwork()
    net(tf.random_uniform((1, 2), minval=-1, maxval=1))