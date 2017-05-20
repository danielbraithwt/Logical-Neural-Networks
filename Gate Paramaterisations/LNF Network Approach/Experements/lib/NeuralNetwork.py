import sys
import time
import tensorflow as tf
import numpy as np

def noisy_or_activation(tensor):
    return 1 - tf.exp(-tensor)

def noisy_and_activation(tensor):
    return tf.exp(-tensor)

def perceptron_activation(tensor):
    return 1 / (1 + tf.exp(-tensor))

def train_network(N, data, targets, hiddenActivation, outputActivation, minWeightValue):
    with tf.device("/cpu:0"):
        ones = tf.constant(np.repeat([1.0], 2**N).T, name='ones')

        # Data and target variables
        x = tf.placeholder("float64", [2**N, N])
        y = tf.placeholder("float64", [2**N])

        x_prime = tf.concat([tf.expand_dims(ones, 1), x], axis=1)
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.rand(2**N, N + 1)), name='w_hidden')
        w_out = tf.Variable(np.random.rand(2**N + 1), name='w_out')

        # Compute output of hidden layer
        hidden_out = 1 - tf.exp(-tf.matmul(w_hidden, tf.transpose(x_prime)))
        hidden_out_prime = tf.concat([tf.expand_dims(ones, 1), hidden_out], axis=1)

        # Compute output of network
        y_hat = tf.exp(-tf.matmul(hidden_out_prime, tf.expand_dims(w_out, 1)))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(tf.transpose(y), 1) - y_hat))

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)

        clip_op_hidden = tf.assign(w_hidden, tf.clip_by_value(w_hidden, minWeightValue, np.infty))
        clip_op_out = tf.assign(w_out, tf.clip_by_value(w_out, minWeightValue, np.infty))

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        for i in range(10000):
            session.run(train_op, feed_dict={x:data, y:targets})
            session.run(clip_op_hidden)
            session.run(clip_op_out)

        final_error = session.run(error, feed_dict={x:data, y:targets})
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)

    total_time = time.time() - start_time

    return (w_hidden_value, w_out_value), final_error, total_time
