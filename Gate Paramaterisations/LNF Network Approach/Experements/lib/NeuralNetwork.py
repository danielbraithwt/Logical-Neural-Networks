import sys
import time
import tensorflow as tf
import numpy as np
import Pruning

def noisy_or_activation(iput, weights, s):
    iput_prime = tf.concat([tf.expand_dims(np.repeat([1.0], s), 1), iput], axis=1)
    z = tf.matmul(iput_prime, weights)

    return 1 - tf.exp(-z)

def noisy_and_activation(iput, weights, s):
    iput_prime = tf.concat([tf.expand_dims(np.repeat([1.0], s), 1), 1-iput], axis=1)
    z = tf.matmul(iput_prime, weights)
    
    return tf.exp(-z)

def perceptron_activation(iput, weights, s):
    iput_prime = tf.concat([tf.expand_dims(np.repeat([1.0], s), 1), iput], axis=1)
    z = tf.matmul(iput_prime, weights)
    
    return 1 / (1 + tf.exp(-z))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def compute_weight_penalty(weights):
    return tf.reduce_sum(tf.exp(-0.8 * tf.sqrt(tf.pow(weights - 5, 2))))#tf.reduce_sum(tf.exp(-4.5 * tf.abs(weights - 1.5)))

def run_network(N, net, data, targets, hiddenActivation, outputActivation):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None, None])
        y = tf.placeholder("float64", [None])
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.uniform(0, 1, (2**N, 2*N + 1))), name='w_hidden')
        w_out = tf.Variable(np.random.uniform(0, 1, 2**N + 1), name='w_out')

        # Compute output of hidden layer
        hidden_out = hiddenActivation(x, tf.transpose(w_hidden), len(data))

        # Compute output of network
        y_hat = outputActivation(hidden_out, tf.expand_dims(w_out, 1), len(data))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))
        
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)

        final_error = session.run(error, feed_dict={x:data, y:targets})


    return final_error


def train_network(N, data, targets, hiddenActivation, outputActivation, minWeightValue, iterations, lr):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None, None])
        y = tf.placeholder("float64", [None])
        r = tf.placeholder("float64")
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.uniform(0, 10, (2**N, 2*N + 1))), name='w_hidden')
        w_out = tf.Variable(np.random.uniform(0, 10, 2**N + 1), name='w_out')

        # Compute output of hidden layer
        hidden_out = hiddenActivation(x, tf.transpose(w_hidden), len(data))

        # Compute output of network
        y_hat = outputActivation(hidden_out, tf.expand_dims(w_out, 1), len(data))

        penalty = r * (compute_weight_penalty(w_hidden) + compute_weight_penalty(w_out))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))
        loss = error + penalty * 0.0

        train_op = tf.train.AdamOptimizer().minimize(loss)#tf.train.GradientDescentOptimizer(lr).minimize(loss)

        clip_op_hidden = tf.assign(w_hidden, tf.clip_by_value(w_hidden, minWeightValue, 10))
        clip_op_out = tf.assign(w_out, tf.clip_by_value(w_out, minWeightValue, 10))

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            session.run(train_op, feed_dict={x:data, y:targets, r: float(i)/float(iterations)})
            session.run(clip_op_hidden)
            session.run(clip_op_out)

            if i % 100 == 0:
                print("Loss: ", session.run(loss, feed_dict={x:data, y:targets, r: float(i)/float(iterations)}))
                print("\tError: ", session.run(error, feed_dict={x:data, y:targets, r: float(i)/float(iterations)}))
                print("\tPenalty: ", session.run(penalty, feed_dict={r: float(i)/float(iterations)}))
                print()

        error = session.run(error, feed_dict={x:data, y:targets})
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)

    total_time = time.time() - start_time

    return (w_hidden_value, w_out_value), error, total_time


def train_perceptron_network_general(N, data, targets, iterations, lr):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None, None])
        y = tf.placeholder("float64", [None])
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.rand(N, 2*N + 1)), name='w_hidden')
        w_l1 = tf.Variable(np.random.rand(N, N + 1), name='w_l1')
        w_out = tf.Variable(np.random.rand(1, N + 1), name='w_l1')

        # Compute output of hidden layer
        hidden_out = perceptron_activation(x, tf.transpose(w_hidden), len(data))

        l1_out = perceptron_activation(hidden_out, tf.transpose(w_l1), len(data))

        # Compute output of network
        y_hat = perceptron_activation(l1_out, tf.transpose(w_out), len(data))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(error)

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

    return (w_hidden_value, w_out_value), error, total_time

def run_perceptron_network_general(N, net, data, targets):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None, None])
        y = tf.placeholder("float64", [None])
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.rand(N, 2*N + 1)), name='w_hidden')
        w_l1 = tf.Variable(np.random.rand(N, N + 1), name='w_l1')
        w_out = tf.Variable(np.random.rand(1, N + 1), name='w_l1')

        # Compute output of hidden layer
        hidden_out = perceptron_activation(x, tf.transpose(w_hidden), len(data))

        l1_out = perceptron_activation(hidden_out, tf.transpose(w_l1), len(data))

        # Compute output of network
        y_hat = perceptron_activation(l1_out, tf.transpose(w_out), len(data))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))
        
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)

        final_error = session.run(error, feed_dict={x:data, y:targets})


    return final_error






def train_network_for_loss(N, data, targets, hiddenActivation, outputActivation, minWeightValue, iterations, lr, interval):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None, None])
        y = tf.placeholder("float64", [None])
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.uniform(low=0.0, high=1.0, size=(2**N, 2*N + 1))), name='w_hidden')
        w_out = tf.Variable(np.random.uniform(low=0.0, high=1.0, size=(2**N + 1)), name='w_out')

        # Compute output of hidden layer
        hidden_out = hiddenActivation(x, tf.transpose(w_hidden), len(data))

        # Compute output of network
        y_hat = outputActivation(hidden_out, tf.expand_dims(w_out, 1), len(data))

        # Compute error
        error = tf.reduce_sum(tf.square(tf.expand_dims(y, 1) - y_hat))

        train_op = tf.train.RMSPropOptimizer(lr).minimize(error)

        clip_op_hidden = tf.assign(w_hidden, tf.clip_by_value(w_hidden, minWeightValue, np.infty))
        clip_op_out = tf.assign(w_out, tf.clip_by_value(w_out, minWeightValue, np.infty))


    model = tf.global_variables_initializer()
    losses = []

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):
            if i % interval == 0:
                print(i)
                losses.append(session.run(error, feed_dict={x:data, y:targets}))
                
            session.run(train_op, feed_dict={x:data, y:targets})
            session.run(clip_op_hidden)
            session.run(clip_op_out)

        error = session.run(error, feed_dict={x:data, y:targets})
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)

        print(error)

    return losses
