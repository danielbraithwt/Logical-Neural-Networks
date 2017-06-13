import sys
import time
import tensorflow as tf
import random
import numpy as np

def noisy_or_activation(iput, weights, s):
    iput_prime = tf.concat([[1.0], iput], axis=0)
    apply_weights = lambda x: 1 - tf.reduce_prod(tf.pow(x, iput_prime))
    return tf.map_fn(apply_weights, weights)
    #return 1 - tf.reduce_prod(z, axis=1)

    #return 1 - tf.exp(-z)

def noisy_and_activation(iput, weights, s):
    iput_prime = tf.concat([[1.0], 1-iput], axis=0)
    #apply_weights = lambda x: tf.reduce_prod(tf.pow(x, iput_prime))
    return tf.reduce_prod(tf.pow(weights, iput_prime))
    #return tf.reduce_prod(z)
    #iput_prime = tf.concat([tf.expand_dims(np.repeat([1.0], s), 1), 1-iput], axis=1)
    #z = tf.matmul(iput_prime, weights)
    
    #return tf.exp(-z)

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

def train_network(N, data, target, iterations, lr):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None])
        y = tf.placeholder("float64", )
            
        # Set up weights
        w_hidden = tf.Variable(np.array(np.random.rand(2**N, 2*N + 1)), name='w_hidden')
        w_out = tf.Variable(np.random.rand(2**N + 1), name='w_out')

        # Compute output of hidden layer
        hidden_out = noisy_or_activation(x, w_hidden, 0)

        # Compute output of network
        y_hat = noisy_and_activation(hidden_out, w_out, 0)

        # Compute error
        y_hat_prime = tf.clip_by_value(y_hat, 0.00000001, 1.0)
        error = -(y * tf.log(y_hat_prime) + (1-y) * tf.log(1-y_hat_prime))#tf.pow(y - y_hat, 2)

        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

        clip_op_hidden = tf.assign(w_hidden, tf.clip_by_value(w_hidden, 0.00000001, 1))
        clip_op_out = tf.assign(w_out, tf.clip_by_value(w_out, 0.00000001, 1))

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        #print(session.run(y_hat, feed_dict={x:data[0], y:targets[0]}))
        for i in range(iterations):
            if i % 10 == 0:
                e = 0
                for d in range(len(data)):
                    er = session.run(error, feed_dict={x:data[d], y:targets[d]})
                    e += er

                print(e)
                print(session.run(w_hidden))
                print()
            
            for d in range(len(data)):
                session.run(train_op, feed_dict={x:data[d], y:targets[d]})
                session.run(clip_op_hidden)
                session.run(clip_op_out)

        #error = session.run(error, feed_dict={x:data, y:targets})
        
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)

    total_time = time.time() - start_time

    return (w_hidden_value, w_out_value), 0, total_time


def __perms(n):
    if not n:
        return

    p = []

    for i in range(0, 2**n):
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p

def __n_rand_perms(n, size):
    if not n:
        return

    idx = [random.randrange(2**n) for i in range(size)]

    p = []

    for i in idx:
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p

def generateExpressions(n):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), 1)

    return np.array(list(map(lambda x: (inputs, x), outputs)))

n = 2
if __name__ == '__main__':
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    network, loss, time = train_network(n, data, targets, 50000, 0.003)
