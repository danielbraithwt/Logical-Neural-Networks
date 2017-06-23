import sys
import time
import tensorflow as tf
import random
import numpy as np

def noisy_or_activation(iput, weights):
    z = tf.matmul(iput, tf.transpose(weights))
    return 1 - tf.exp(-z)

def noisy_and_activation_part1(iput, weights):
    return tf.matmul(1-iput, tf.transpose(weights))

def noisy_and_activation_part2(t):
    return tf.exp(-t)

def add_ones(tensor, s):
    #print(tensor.shape)
    ones = np.expand_dims(np.repeat([1.0], s), 1)
    return np.concatenate([ones, tensor], axis=1)

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def compute_weight_penalty(weights, lim):
    return tf.reduce_sum(tf.exp(-0.04 * tf.pow(weights - (lim/2), 2)))

def train_network(N, data, targets, iterations, lr):
    UPPER_LIM = -np.log(0.00000001)
    data = add_ones(np.array(list(map(lambda x: transform_input(x), data))), len(data)).tolist()

    # Create weight mask
    m = np.concatenate((np.zeros((1, 2*N), dtype='float64'), np.identity(2*N, dtype='float64')))
    mask = tf.constant(m)
    
    # Data and target variables
    x = tf.constant(data, dtype='float64')#tf.placeholder("float64", [None, None])
    y = tf.constant([targets.tolist()], dtype='float64')#tf.placeholder("float64", )
    #r = tf.placeholder("float64")

    w_hidden = tf.Variable(np.array(np.random.uniform(0.0, UPPER_LIM, (2**N, 2*N + 1))), name='w_hidden', dtype='float64')
    w_out = tf.Variable(np.random.uniform(0.0, UPPER_LIM, (1, 2**N)), name='w_out', dtype='float64')
    b_out = tf.Variable(np.random.uniform(0.0, UPPER_LIM), dtype='float64')

    r_w_hidden = tf.nn.relu(w_hidden)
    r_w_out = tf.nn.relu(w_out)
    r_b_out = tf.nn.relu(b_out)

    #with tf.device('/cpu:0'):
    hidden_out = noisy_or_activation(x, r_w_hidden)
    y_hat = noisy_and_activation_part2(noisy_and_activation_part1(hidden_out, r_w_out) + r_b_out)
    y_hat_prime = tf.transpose(y_hat)

    w_hidden_penalty = compute_weight_penalty(tf.matmul(r_w_hidden, mask), UPPER_LIM)
    w_out_penalty = compute_weight_penalty(r_w_out, UPPER_LIM)
    penalty = 0.000001 * (w_hidden_penalty + w_out_penalty)

    errors = tf.pow(y - y_hat_prime, 2)
    error = tf.reduce_sum(errors)

    loss = error #+ penalty

    train_op_error = tf.train.AdamOptimizer(0.0003).minimize(error)
    train_op_penalty = tf.train.AdamOptimizer(0.0003).minimize(penalty)
    #train_op = tf.train.AdamOptimizer(0.0003).minimize(loss)

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)
        #print(session.run(r_w_hidden))
        #print(session.run(error))
        #print(session.run([train_op_error, train_op_penalty]))
        #print(session.run(error))
        for i in range(iterations):
            session.run([train_op_error, train_op_penalty])
            #session.run(train_op)

            if i % 10 == 0:
                l, e, p = session.run([loss, error, penalty])
                print("Loss: ", l)
                print("\tError: ", e)
                print("\tPenalty: ", p)
                print()

        print(session.run(w_hidden))
        print(session.run(w_out))
        

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
np.random.seed(1234)
if __name__ == '__main__':
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    print(targets)

    network, r_net, loss, time = train_network(n, data, targets, 30000, 0.003)
    print(network)
    #print(data)
    print(targets)
