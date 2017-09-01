from tensorflow.contrib import layers
from config import *
from layer import *
import tensorflow as tf

class GlimpsNetwork(object):
    def __init__(self, location_ph):
        self.location_ph = location_ph

    def getGlimps(self, loc_tensor):
        self.glimps_imgs = tf.reshape(self.location_ph, [-1, 28, 28, 1], name='reshape_layer_1')
        self.glimps_imgs = tf.image.extract_glimpse(self.glimps_imgs, [win_size, win_size], loc_tensor)
        self.glimps_imgs = tf.reshape(self.glimps_imgs, [-1, win_size * win_size * 1])
        return self.glimps_imgs

    def __call__(self, loc_tensor):
        self.retina_imgs = self.getGlimps(loc_tensor)
        self.retina_imgs = tf.nn.relu(Dense(self.retina_imgs, n_units=128, name='glimps_ind_fc_1'))
        self.location_net = tf.nn.relu(Dense(loc_tensor, n_units=128, name='glimps_ind_fc_2'))
        self.glimps_net = tf.concat([self.retina_imgs, self.location_net], axis=-1)
        self.glimps_net = tf.nn.relu(Dense(self.glimps_net, n_units=256, name='glimps_merge_fc_1'))
        return self.glimps_net

class LocationNetwork(object):
    def __call__(self, state_tensor):
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            self.location_net = Dense(state_tensor, n_units=loc_dim, name='location_net_fc1')
            self.mean = tf.stop_gradient(tf.clip_by_value(self.location_net, -1.0, 1.0))
            self.location = self.mean + tf.random_normal((tf.shape(state_tensor)[0], loc_dim), stddev=loc_std)
            self.location = tf.stop_gradient(self.location)
            return self.location, self.mean

if __name__ == '__main__':
    location_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    net = LocationNetwork()
    net(tf.random_uniform((1, 2), minval=-1, maxval=1))