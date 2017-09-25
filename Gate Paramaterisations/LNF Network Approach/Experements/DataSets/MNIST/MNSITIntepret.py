import sys
sys.path.append('../../lib/')

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def noisy_or_activation(inputs, weights):
    t_w = weights

    z = tf.matmul(inputs, tf.transpose(t_w))
    return 1 - tf.exp(-z)

def noisy_and_activation(inputs, weights):
    t_w = weights

    z = tf.matmul(1 - inputs, tf.transpose(t_w))
    return tf.exp(-z)

def transform_weights(weights):
    return np.log(1 + np.exp(-weights))

activations = [noisy_or_activation, noisy_and_activation, noisy_and_activation]
network = np.load('network.npy')

# Format network model
#formatted_model = []
#for l in network:
#    formatted_model.append(sigmoid(l[0]))


target_output_pattern = tf.constant([1,0,0,0,0,0,0,0,0,0], dtype='float32')
y = target_output_pattern

layers = []
for i in range(len(network)):
    weights = transform_weights(network[i][0])
    #print(weights)
    net_weights = tf.constant(weights, dtype='float32')
    layers.append(net_weights)

input_pattern = tf.Variable(np.random.uniform(0.4, 0.6, (1, 28 * 28)), dtype='float32')

output_pattern = input_pattern
for l in range(len(layers)):
    output_pattern = tf.concat([output_pattern, 1 - output_pattern], axis=1)
    output_pattern = activations[i](output_pattern, layers[l])

    
#output_pattern = output_pattern * (1/tf.reduce_sum(output_pattern))
y_hat_prime = output_pattern
y_hat_prime_0 = tf.clip_by_value(y_hat_prime, 1e-20, 1)
y_hat_prime_1 = tf.clip_by_value(1 - y_hat_prime, 1e-20, 1)
errors = -(y * tf.log(y_hat_prime_0) + (1-y) * tf.log(y_hat_prime_1))#
error = tf.reduce_sum(errors)

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)



model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(10000):
        print(session.run(output_pattern))
        print(session.run(error))
        print()
        session.run(train_op)

