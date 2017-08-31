from tensorlayer.layers import Layer
from config import *
import tensorlayer as tl
import tensorflow as tf

class GlimpsExtractLayer(Layer):
  def __init__(
      self,
      layer = None,
      loc_tensor = None,
      act = tf.nn.relu,
      name ='simple_dense',
  ):
      # check layer name (fixed)
      Layer.__init__(self, name=name)

      # the input of this layer is the output of previous layer (fixed)
      self.inputs = layer.outputs

      # operation (customized)
      n_in = int(self.inputs._shape[-1])
      self.loc_tensor = loc_tensor
      with tf.variable_scope(name) as vs:

          # tensor operation
          self.outputs = tf.image.extract_glimpse(self.inputs, [win_size, win_size], self.loc_tensor)

      # get stuff from previous layer (fixed)
      self.all_layers = list(layer.all_layers)
      self.all_params = list(layer.all_params)
      self.all_drop = dict(layer.all_drop)

      # update layer (customized)
      self.all_layers.extend( [self.outputs] )