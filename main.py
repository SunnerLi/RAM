from tensorflow.contrib.legacy_seq2seq import rnn_decoder
from network import GlimpsNetwork, LocationNetwork
from tensorflow.examples.tutorials import mnist
from config import *
import tensorflow as tf
import numpy as np

# List object to record coordinate
origin_coor_list = []
sample_coor_list = []

# Network object
location_network = None
glimps_network = None

def getNextRetina(output, i):
    global origin_coor_list
    global sample_coor_list
    global location_network
    global glimps_network

    #print(location_network.location_net)
    
    sample_coor, origin_coor = location_network(output)
    
    origin_coor_list.append(origin_coor)
    sample_coor_list.append(sample_coor)
    return glimps_network(sample_coor)

if __name__ == '__main__':
    # Create placeholder
    images_ph = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
    labels_ph = tf.placeholder(tf.int64, [None])

    # Create network
    glimps_network = GlimpsNetwork(images_ph)
    location_network = LocationNetwork()
    
    # Construct Glimps network (part in core network)
    init_location = tf.random_uniform((batch_size, 2), minval=-1.0, maxval=1.0)
    init_glimps_tensor = glimps_network(init_location)

    # Construct core network
    lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
    init_lstm_state = lstm_cell.zero_state(batch_size, tf.float32)
    input_glimps_tensor_list = [init_glimps_tensor]
    input_glimps_tensor_list.append([0] * num_glimpses)
    outputs, _ = rnn_decoder(input_glimps_tensor_list, init_lstm_state, lstm_cell, loop_function=getNextRetina)