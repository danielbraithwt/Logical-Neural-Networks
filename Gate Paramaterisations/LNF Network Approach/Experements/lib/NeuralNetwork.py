import sys
import time
import tensorflow as tf
import numpy as np
import Pruning
import random

def perceptron_activation(iput, weights, bias):
    z = tf.add(tf.matmul(iput, weights), bias)
    
    return 1 / (1 + tf.exp(-z))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

##def run_network(N, net, data, targets):
##    # Convert input to the new form with the not of atoms
##    data = list(map(lambda x: transform_input(x), data))
##    
##    with tf.device("/cpu:0"):
##        # Data and target variables
##        x = tf.placeholder("float64", [None, None])
##        y = tf.placeholder("float64", [None])
##            
##        # Set up weights
##        w_hidden = tf.Variable(np.array(np.random.uniform(0, 1, (2**N, 2*N + 1))), name='w_hidden')
##        w_out = tf.Variable(np.random.uniform(0, 1, 2**N + 1), name='w_out')
##
##        # Compute output of hidden layer
##        hidden_out = hiddenActivation(x, tf.transpose(w_hidden), len(data))
##
##        # Compute output of network
##        y_hat = outputActivation(hidden_out, tf.expand_dims(w_out, 1), len(data))
##
##        # Compute error
##        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))
##        
##    model = tf.global_variables_initializer()
##
##    with tf.Session() as session:
##        session.run(model)
##
##        final_error = session.run(error, feed_dict={x:data, y:targets})
##
##
##    return final_error


def train_perceptron_network(N, data, targets, iterations):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float32", [None, None])
        y = tf.placeholder("float32", )
            
        # Set up weights
        w_hidden = tf.Variable(np.random.rand(2**N, 2*N), dtype='float32')
        b_hidden = tf.Variable(np.random.rand(1, 2**N), dtype='float32')
        
        w_out = tf.Variable(np.random.rand(1, 2**N), dtype='float32')
        b_out = tf.Variable(np.random.rand(1), dtype='float32')

        # Compute output of hidden layer
        hidden_out = perceptron_activation(x, tf.transpose(w_hidden), b_hidden)

        # Compute output of network
        y_hat = perceptron_activation(hidden_out, tf.transpose(w_out), b_out)

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            session.run(train_op, feed_dict={x:data, y:targets})


        error = session.run(error, feed_dict={x:data, y:targets})
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)

    total_time = time.time() - start_time

    return None, (w_hidden_value, w_out_value), error, total_time


def train_perceptron_network_general(N, data, targets, iterations, num_out=1):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float32", [None, None])
        y = tf.placeholder("float32", )
            
        # Set up weights
        w_hidden = tf.Variable(np.random.rand(N, 2*N), dtype='float32')
        b_hidden = tf.Variable(np.random.rand(1, N), dtype='float32')
        
        #w_l1 = tf.Variable(np.random.rand(N, N), dtype='float32')
        #b_l1 = tf.Variable(np.random.rand(1, N), dtype='float32')
        
        w_out = tf.Variable(np.random.rand(num_out, N), dtype='float32')
        b_out = tf.Variable(np.random.rand(num_out), dtype='float32')

        # Compute output of hidden layer
        hidden_out = perceptron_activation(x, tf.transpose(w_hidden), b_hidden)

        l1_out = hidden_out#perceptron_activation(hidden_out, tf.transpose(w_l1), b_l1)

        # Compute output of network
        y_hat = perceptron_activation(l1_out, tf.transpose(w_out), b_out)
        #y_hat = sigmoid(tf.add(tf.matmul(l1_out, tf.transpose(w_out)), b_out))
        #y_hat_prime = tf.nn.softmax(y_hat)
        y_prime = tf.expand_dims(y, 1)#tf.nn.softmax(tf.transpose(y))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))#tf.nn.softmax_cross_entropy_with_logits(labels=y_prime, logits=y_hat)#tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))

        train_op = tf.train.GradientDescentOptimizer(0.02).minimize(error)

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            session.run(train_op, feed_dict={x:data, y:targets})

            #if i % 100 == 0:
            #    print(session.run(error, feed_dict={x:data, y:targets}))

        error = session.run(error, feed_dict={x:data, y:targets})
        w_hidden_value = session.run(w_hidden)
        b_hidden_value = session.run(b_hidden)

        #w_l1_value = session.run(w_l1)
        #b_l1_value = session.run(b_l1)
        
        w_out_value = session.run(w_out)
        b_out_value = session.run(b_out)

    total_time = time.time() - start_time

    #return None, ((w_hidden_value, b_hidden_value), (w_l1_value, b_l1_value), (w_out_value, b_out_value)), error, total_time
    return None, ((w_hidden_value, b_hidden_value), (w_out_value, b_out_value)), error, total_time


def run_perceptron_network_general(N, data, targets, net):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float32", [None, None])
        y = tf.placeholder("float32", )
            
        # Set up weights
        w_hidden = tf.Variable(net[0][0], dtype='float32')
        b_hidden = tf.Variable(net[0][1], dtype='float32')
        
        w_l1 = tf.Variable(net[1][0], dtype='float32')
        b_l1 = tf.Variable(net[1][1], dtype='float32')
        
        w_out = tf.Variable(net[2][0], dtype='float32')
        b_out = tf.Variable(net[2][1], dtype='float32')

        # Compute output of hidden layer
        hidden_out = perceptron_activation(x, tf.transpose(w_hidden), b_hidden)

        l1_out = perceptron_activation(hidden_out, tf.transpose(w_l1), b_l1)

        # Compute output of network
        y_hat = perceptron_activation(l1_out, tf.transpose(w_out), b_out)

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))
        
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)

        final_error = session.run(error, feed_dict={x:data, y:targets})


    return final_error
