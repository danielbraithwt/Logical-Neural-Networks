import sys
import time
import tensorflow as tf
import random
import numpy as np

def noisy_or_activation(iput, weights, s):
    iput_prime = tf.concat([[1.0], iput], axis=0)
    apply_weights = lambda x: 1 - tf.reduce_prod(tf.pow(x, iput_prime))
    return tf.map_fn(apply_weights, weights)

def noisy_and_activation(iput, weights, s):
    iput_prime = tf.concat([[1.0], 1-iput], axis=0)
    return tf.reduce_prod(tf.pow(weights, iput_prime))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def compute_weight_penalty(weights, mask):
    #clip_byas = lambda x: tf.multiply(x, tf.concat([np.array([0.0], dtype="float64"), np.repeat(1.0, ones)], axis=0))
    #weights_prime = tf.map_fn(clip_byas, weights)
    #return weights_prime
    #return mask
    weights_prime = tf.matmul(weights, mask)
    return tf.reduce_sum(tf.exp(-25.0 * tf.pow(weights_prime - 0.5, 2)))

def train_network(N, data, targets, iterations, lr):
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: transform_input(x), data))
    num_hidden = int(float((2**N))/2.0)

    m1 = tf.constant(np.concatenate((np.zeros((1, 2*N), dtype='float64'), np.identity(2*N, dtype='float64')), axis=0).tolist(), dtype='float64')
    m2 = tf.constant(np.concatenate((np.zeros((1, num_hidden), dtype='float64'), np.identity(num_hidden, dtype='float64')), axis=0).tolist(), dtype='float64')
    
    # Data and target variables
    x = tf.placeholder("float64", [None])
    y = tf.placeholder("float64", )
    #r = tf.placeholder("float64")
        
    # Set up weights
    w_hidden = tf.Variable(np.array(np.random.uniform(0.0, 1.0, (num_hidden, 2*N + 1))), name='w_hidden')
    w_out = tf.Variable(np.random.uniform(0.0, 1.0, num_hidden + 1), name='w_out')

    r_w_hidden = tf.round(w_hidden)
    r_w_out = tf.round(w_out)

    
    # Compute output of hidden layer
    hidden_out = noisy_or_activation(x, w_hidden, 0)

    # Compute output of network
    y_hat = noisy_and_activation(hidden_out, w_out, 0)
    y_hat_prime = tf.clip_by_value(y_hat, 0.00000001, 1)

    # Compute error
    penalty = 0.025 * (compute_weight_penalty(w_hidden, m1) + compute_weight_penalty(tf.transpose(tf.expand_dims(w_out,1)), m2))

    clip_op_hidden = tf.assign(w_hidden, tf.clip_by_value(w_hidden, 0.00000001, 1))
    clip_op_out = tf.assign(w_out, tf.clip_by_value(w_out, 0.00000001, 1))
    error = tf.pow(y - y_hat_prime, 2)

    #loss = error + penalty * 0.09

    #train_op_error = tf.train.GradientDescentOptimizer(0.0001).minimize(error)
    #train_op_penalty = tf.train.GradientDescentOptimizer(0.0001).minimize(penalty)

    train_op_error = tf.train.AdamOptimizer(0.0003).minimize(error)
    train_op_penalty = tf.train.AdamOptimizer(0.0003).minimize(penalty)

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        #print(session.run(hidden_out, feed_dict={x:data[0], y:targets[0]}))
        #print(session.run(penalty))
        #return
        for i in range(iterations):   
            for d in range(len(data)):
                session.run(train_op_error, feed_dict={x:data[d], y:targets[d]})
                session.run(clip_op_hidden)
                session.run(clip_op_out)

            session.run(train_op_penalty)
            session.run(clip_op_hidden)
            session.run(clip_op_out)

            if i % 100 == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:data[d], y:targets[d]})

                #print(session.run(w_hidden))
                print(er)
                print(session.run(penalty))
                #print(session.run(w_hidden))
                #print(session.run(w_out))
                print()

        #error = session.run(error, feed_dict={x:data, y:targets})
        
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)

        r_w_h = session.run(r_w_hidden)
        r_w_o = session.run(r_w_out)

    total_time = time.time() - start_time

    return (w_hidden_value, w_out_value), (r_w_h, r_w_o), 0, total_time


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

n = 3
if __name__ == '__main__':
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    print(targets)

    network, r_net, loss, time = train_network(n, data, targets, 10000, 0.003)
    print(network)
    #print(data)
    print(targets)
