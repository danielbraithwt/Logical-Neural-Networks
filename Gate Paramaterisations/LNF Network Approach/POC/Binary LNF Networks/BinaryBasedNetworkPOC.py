import sys
import time
import tensorflow as tf
from tensorflow.python.client import timeline
import random
import numpy as np

def noisy_or_activation(iput, weights, bias):
    zero = tf.constant(0.0, dtype="float64")
    one = tf.constant(1.0, dtype="float64")

    z = tf.add(tf.matmul(iput, weights), bias)
    return tf.subtract(one, tf.exp(tf.subtract(zero, z)))

def noisy_and_activation(iput, weights, bias):
    zero = tf.constant(0.0, dtype="float64")
    one = tf.constant(1.0, dtype="float64")

    z = tf.add(tf.matmul(tf.subtract(one, iput), weights), bias)
    return tf.exp(tf.subtract(zero, z))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def compute_weight_penalty(weights):
    return tf.reduce_sum(tf.exp(tf.multiply(tf.constant(-0.05, dtype="float64"), tf.pow(tf.subtract(weights, tf.constant(9.21034, dtype="float64")), 2))))
    #return tf.reduce_sum(tf.exp(tf.multiply(tf.constant(-25.0, dtype="float64"), tf.pow(tf.subtract(weights, tf.constant(0.5, dtype="float64")), 2))))

def train_network(N, data, targets, iterations, lr):    
    # Convert input to the new form with the not of atoms
    data = list(map(lambda x: np.transpose(np.expand_dims(transform_input(x), 1)), data))
    #data = add_ones(data, len(data))

    # Data and target variables
    x = tf.placeholder("float64", [None, None])
    y = tf.placeholder("float64")
    r = tf.placeholder("float64")

    with tf.device('/cpu:0'):
        upper = -np.log(0.00000001)
        
        zero = tf.constant(0.0, dtype="float64")
        one = tf.constant(1.0, dtype="float64")

        w_hidden = tf.Variable(np.array(np.random.uniform(0.0, upper, (2**N, 2*N))), dtype="float64")
        b_hidden = tf.Variable(np.transpose(np.random.uniform(0.0, upper, (2**N, 1))), dtype="float64")
        
        w_out = tf.Variable(tf.transpose(np.expand_dims(np.random.uniform(0.0, upper, 2**N), 1)), dtype="float64")
        b_out = tf.Variable(np.random.uniform(0.0, upper, (1)), dtype="float64")

        c_w_hidden = tf.clip_by_value(w_hidden, 0, upper)
        c_b_hidden = tf.clip_by_value(b_hidden, 0, upper)
        
        c_w_out = tf.clip_by_value(w_out, 0, upper)
        c_b_out = tf.clip_by_value(b_out, 0, upper)

        #c_w_hidden = tf.clip_by_value(w_hidden, 0.00000001, 1)
        #c_b_hidden = tf.clip_by_value(b_hidden, 0.00000001, 1)
        
        #c_w_out = tf.clip_by_value(w_out, 0.00000001, 1)
        #c_b_out = tf.clip_by_value(b_out, 0.00000001, 1)
        
        #t_w_hidden = tf.subtract(zero, tf.log(c_w_hidden))
        #t_b_hidden = tf.subtract(zero, tf.log(c_b_hidden))
                        
        #t_w_out = tf.subtract(zero, tf.log(c_w_out))
        #t_b_out = tf.subtract(zero, tf.log(c_b_out))

        
        hidden_out = noisy_or_activation(x, tf.transpose(c_w_hidden), c_b_hidden)

        y_hat = noisy_and_activation(hidden_out, tf.transpose(c_w_out), c_b_out)

        penalty = tf.add(compute_weight_penalty(c_w_hidden), compute_weight_penalty(c_w_out))

    y_hat_prime = tf.reduce_sum(y_hat)
    error = -(y * tf.log(y_hat_prime) + (1-y) * tf.log(1-y_hat_prime))#tf.pow(y - tf.reduce_sum(y_hat), 2)

    e = error + 0.002 * penalty
    train_op_error = tf.train.AdamOptimizer(0.003).minimize(error)
    train_op_penalty = tf.train.AdamOptimizer(0.003).minimize(penalty)
    #train_op = tf.train.AdamOptimizer(lr).minimize(e)
    r_w_hidden = tf.round(tf.exp(-c_w_hidden))
    r_w_out = tf.round(tf.exp(-c_w_out))

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session() as session:
        session.run(model)
        for i in range(iterations):            
            for d in range(len(data)):
                session.run([train_op_error], feed_dict={x:data[d], y:targets[d]})
            session.run(train_op_penalty)

            if i % 100 == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:data[d], y:targets[d]})

                print(i, " : ", iterations)
                print(er)
                print(session.run(penalty, feed_dict={r: float(i)/float(iterations)}))
                print()
        
        w_hidden_value = session.run(c_w_hidden)
        w_out_value = session.run(c_w_out)

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

n = 4
if __name__ == '__main__':
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    print(targets)

    network, r_net, loss, time = train_network(n, data, targets, 15000, 0.006)
    print(network)
    #print(data)
    print(targets)
