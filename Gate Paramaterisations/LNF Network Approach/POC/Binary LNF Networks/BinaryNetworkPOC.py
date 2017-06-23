import sys
import time
import tensorflow as tf
import random
import numpy as np

def noisy_or_activation(iput, weights, bias):  
    apply_weights = lambda x: tf.reduce_prod(tf.pow(x, iput))
    return 1 - tf.multiply(tf.map_fn(apply_weights, weights), bias)

def noisy_and_activation(iput, weights, bias):
    return tf.multiply(tf.reduce_prod(tf.pow(weights, 1 - iput)), bias)

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def add_ones(tensor, s):
    #print(tensor.shape)
    ones = np.expand_dims(np.repeat([1.0], s), 1)
    return np.concatenate([ones, tensor], axis=1)

def add_ones_tf(tensor, s):
    ones = tf.expand_dims(np.repeat([1.0], s), 1)
    return tf.concat([ones, tensor], axis=1)
    
def compute_weight_penalty(weights):
    return tf.reduce_sum(tf.exp(-25.0 * tf.pow(weights - 0.5, 2)))

def train_network(N, data, targets, iterations, lr):
    # Convert input to the new form with the not of atoms
    data = add_ones(np.array(list(map(lambda x: transform_input(x), data))))
    #examples = tf.constant(data, dtype="float64")
    #outs = tf.constant(targets, dtype="float64")

    # Setup Queue
##    queue = tf.FIFOQueue(capacity=len(data), dtypes=[tf.float64, tf.float64])
##    enqueue_op = queue.enqueue_many([data, targets])
##
##    num_threads = 1
##    x, y = queue.dequeue()
##    
##    qr = tf.train.QueueRunner(queue, [enqueue_op] * num_threads)
##    examples_in_queue = queue.size()
    
    with tf.device("/cpu:0"):
        # Data and target variables
        x = tf.placeholder("float64", [None])
        y = tf.placeholder("float64", )
        #r = tf.placeholder("float64")
            
        # Set up weights
        weights_hidden = tf.Variable(np.array(np.random.uniform(0.1, 0.9, (2**N, 2*N + 1))), name='w_hidden')
        #bias_hidden = tf.Variable(np.array(np.random.uniform(0.1, 0.9, (2**N))), name='b_hidden')
        weights_out = tf.Variable(np.random.uniform(0.1, 0.9, 2**N), name='w_out')
        bias_out = tf.Variable(np.random.uniform(0.1, 0.9, 1), name='b_out')

        
        w_hidden = tf.clip_by_value(tf.nn.relu(weights_hidden), 0.0000000001, 1)
        #b_hidden = tf.clip_by_value(tf.nn.relu(bias_hidden), 0.0000000001, 1)
        w_out = tf.clip_by_value(tf.nn.relu(weights_out), 0.0000000001, 1)
        #b_out = tf.clip_by_value(tf.nn.relu(bias_out), 0.0000000001, 1)
        
        # Compute output of hidden layer
        hidden_out = noisy_or_activation(x, w_hidden, b_hidden)

        # Compute output of network
        y_hat = noisy_and_activation(hidden_out, w_out, b_out)

        # Compute error
        penalty = 0.02 * (compute_weight_penalty(w_hidden) + compute_weight_penalty(w_out))

    
    y_hat_prime = tf.clip_by_value(y_hat, 0.0000000001, 1)

    errors = tf.pow(y - y_hat_prime, 2)
    error = tf.reduce_sum(errors)
    #error = -(y * tf.log(y_hat_prime) + (1-y) * tf.log(1 - y_hat_prime))
    loss = error + penalty

    train_op_error = tf.train.AdamOptimizer(0.0003).minimize(error)
    train_op_penalty = tf.train.AdamOptimizer(0.0003).minimize(penalty)
    train_op = tf.train.AdamOptimizer(0.0003).minimize(loss)

    r_w_hidden = tf.round(w_hidden)
    r_w_out = tf.round(w_out)

    model = tf.global_variables_initializer()

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        session.run(model)

        #coord = tf.train.Coordinator()
        #enqueue_threads = qr.create_threads(session, coord=coord, start=True)
        #print(session.run(hidden_out, feed_dict={x:data[0], y:targets[0]}))
        #return
        for i in range(iterations):
##            if coord.should_stop():
##                break
##
##            err = 0
            for d in range(len(data)):
                #e, _ = session.run([error, train_op_error])
                #err += e
                #print(a, b)
                #print(session.run([examples_in_queue]))
                #session.run(train_op, feed_dict={x:data[d], y:targets[d]})
                session.run(train_op_error, feed_dict={x:data[d], y:targets[d]})
            session.run(train_op_penalty)
##
##            if i % 100 == 0:
##                s = session.run([penalty])
##                print(err, " : ", s)
##            
##            
##            #session.run(train_op_penalty)
##
##        coord.request_stop()
##        coord.join(enqueue_threads)


            if i % 100 == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:data[d], y:targets[d]})

                print(er)
                print(session.run(penalty))
                print()

        #error = session.run(error, feed_dict={x:data, y:targets})
##        
        w_hidden_value = session.run(w_hidden)
        w_out_value = session.run(w_out)
##
        r_w_h = session.run(r_w_hidden)
        r_w_o = session.run(r_w_out)
##
    total_time = time.time() - start_time
##
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

random.seed(1234)
np.random.seed(1234)
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
