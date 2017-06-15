import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math


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

def generate_expressions(n, num):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), num)

    return np.array(list(map(lambda x: (inputs, x), outputs)))

def noisy_or_activation(iput, weights):
    m = lambda x: tf.multiply(iput,x)[0]
    return tf.reduce_max(tf.map_fn(m, weights), axis=1)
    
    #return tf.matmul(iput, weights)

def noisy_and_activation(iput, weights):
    m = lambda x: tf.multiply(iput,x)
    return tf.reduce_min(tf.map_fn(m, weights), axis=1)

def sigmoid(z):
    return 1 / (1 + tf.exp(-z))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

n = 3
in_size = 2 * n
out_size = 1
num_hidden = 2**n

inputs = tf.placeholder('float64', [in_size])
targets = tf.placeholder('float64', [out_size])

hidden_weights = tf.Variable(np.random.uniform(low=0, high=1, size=(num_hidden, in_size)))
output_weights = tf.Variable(np.random.uniform(low=0, high=1, size=(out_size, num_hidden)))

inputs_prime = tf.transpose(tf.expand_dims(inputs, 1))

# Peform Computation
hidden_out = noisy_or_activation(inputs_prime, hidden_weights)

output = noisy_and_activation(hidden_out, output_weights)
errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
error = tf.reduce_sum(errors)

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)
clip_op_hidden = tf.assign(hidden_weights, tf.clip_by_value(hidden_weights, 0, 1))
clip_op_output = tf.assign(output_weights, tf.clip_by_value(output_weights, 0, 1))

model = tf.global_variables_initializer()

points, out = generate_expressions(n, 1)[0]
data = list(map(lambda x: transform_input(x), points))

with tf.Session() as session:
    session.run(model)
    for e in range(6000):
        for d in range(len(data)):
            session.run(train_op, feed_dict={inputs: data[d], targets: [out[d]]})
            session.run(clip_op_hidden)
            session.run(clip_op_output)

        if e % 10 == 0:
            err = 0
            for d in range(len(data)):
                err += session.run(error, feed_dict={inputs: data[d], targets: [out[d]]})
            print(err)

    print(points)
    print(out)
    
    print(session.run(hidden_weights))
    print(session.run(output_weights))
