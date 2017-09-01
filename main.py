from tensorflow.contrib.legacy_seq2seq import rnn_decoder
from tensorflow.contrib import distributions
from network import GlimpsNetwork, LocationNetwork
from tensorflow.examples.tutorials import mnist
from layer import Dense
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

def loglikelihood(mean_arr, sampled_arr, sigma):
    mu = tf.stack(mean_arr)                     # mu = [timesteps, batch_sz, loc_dim]
    sampled = tf.stack(sampled_arr)             # same shape as mu
    gaussian = distributions.Normal(mu, sigma)
    logll = gaussian.log_prob(sampled)           # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    logll = tf.transpose(logll)                 # [batch_sz, timesteps]
    return logll

if __name__ == '__main__':
    # Create placeholder
    images_ph = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
    labels_ph = tf.placeholder(tf.int64, [None])

    # Create network
    glimps_network = GlimpsNetwork(images_ph)
    location_network = LocationNetwork()
    
    # Construct Glimps network (part in core network)
    init_location = tf.random_uniform((tf.shape(images_ph)[0], 2), minval=-1.0, maxval=1.0)
    init_glimps_tensor = glimps_network(init_location)

    # Construct core network
    lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
    init_lstm_state = lstm_cell.zero_state(tf.shape(images_ph)[0], tf.float32)
    input_glimps_tensor_list = [init_glimps_tensor]
    input_glimps_tensor_list.append([0] * num_glimpses)
    outputs, _ = rnn_decoder(input_glimps_tensor_list, init_lstm_state, lstm_cell, loop_function=getNextRetina)

    # Construct the classification network
    logits = Dense(outputs[-1], num_classes, name='classification_net_fc')
    softmax = tf.nn.softmax(logits)

    # Cross-entropy
    entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
    entropy_value = tf.reduce_mean(entropy_value)
    predict_label = tf.argmax(logits, 1)

    # Reward
    reward = tf.cast(tf.equal(predict_label, labels_ph), tf.float32)
    rewards = tf.expand_dims(reward, 1)
    rewards = tf.tile(rewards, (1, num_glimpses))
    _log = loglikelihood(origin_coor_list, sample_coor_list, loc_std)
    _log_ratio = tf.reduce_mean(_log)
    reward = tf.reduce_mean(reward)

    # Hybric locc
    loss = -_log_ratio + entropy_value
    var_list = tf.trainable_variables()
    grads = tf.gradients(loss, var_list)

    # Optimizer
    opt = tf.train.AdamOptimizer(0.0001)
    global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
    train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

    # Train
    with tf.Session() as sess:
        mnist = mnist.input_data.read_data_sets('MNIST_data', one_hot=False)
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            images, labels = mnist.train.next_batch(batch_size)
            images = np.tile(images, [M, 1])
            labels = np.tile(labels, [M])
            _loss_value, _reward_value, _ = sess.run([loss, reward, train_op], feed_dict={
                images_ph: images,
                labels_ph: labels
            })
            if i % 100 == 0:
                print('iter: ', i, '\tloss: ', _loss_value, '\treward: ', _reward_value)