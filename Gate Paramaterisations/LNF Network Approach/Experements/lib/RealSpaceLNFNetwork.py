import sys
import time
import tensorflow as tf
import random
import numpy as np

def noisy_or_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)
    
    z = tf.add(tf.matmul(inputs, tf.transpose(t_w)), t_b)
    return 1 - tf.exp(-z)

def noisy_and_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)

    z = tf.add(tf.matmul(1 - inputs, tf.transpose(t_w)), t_b)
    return tf.exp(-z)

def transform_weights(weights):
    return tf.log((1 + tf.exp(-weights)))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def sigmoid(weights):
    return 1.0/(1 + tf.exp(-weights))

def train_cnf_network(N, data, targets, iterations):
    return _train_network(N, data, targets, iterations, noisy_or_activation, noisy_and_activation)

def train_dnf_network(N, data, targets, iterations):
    return _train_network(N, data, targets, iterations, noisy_and_activation, noisy_or_activation)
    

def _train_network(N, data, targets, iterations, hidden_activation, output_activation):
    data = list(map(lambda x: transform_input(x), data))

    examples = tf.constant(data)
    labels = tf.constant(targets.tolist())

    random_example, random_label = tf.train.slice_input_producer([examples, labels],
                                                           shuffle=False)

    example_batch, label_batch = tf.train.batch([random_example, random_label],
                                          batch_size=1)
    
    # Data and target variables
    x = tf.placeholder("float32", [None, None])
    y = tf.placeholder("float32", )


    w_hidden = tf.Variable(np.random.uniform(-1.0, 1.0, (2**N, 2*N)), dtype='float32')
    b_hidden = tf.Variable(np.transpose(np.random.uniform(-1.0, 1.0, (2**N, 1))), dtype='float32')

    w_out = tf.Variable(np.random.uniform(-1.0, 1.0, (1, 2**N)), dtype='float32')
    b_out = tf.Variable(np.random.uniform(-1.0, 1.0, (1)), dtype='float32')

    hidden_out = hidden_activation(x, w_hidden, b_hidden)
    y_hat = output_activation(hidden_out, w_out, b_out)

    errors = tf.pow(y - y_hat, 2)
    error = tf.reduce_sum(errors)

    train_op_error = tf.train.AdamOptimizer(0.01).minimize(error)

    t_w_hidden = sigmoid(w_hidden)
    t_w_out = sigmoid(w_out)

    r_w_hidden = tf.round(t_w_hidden)
    r_w_out = tf.round(t_w_out)

    model = tf.global_variables_initializer()
    start = time.time()

    with tf.Session() as session:
        session.run(model)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):      
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run([train_op_error], feed_dict={x:batch_ex, y:batch_l})

        er = 0
        for d in range(len(data)):
            er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})

        end = time.time()

        coord.request_stop()
        coord.join(threads)
        
        w_hidden_final = session.run(r_w_hidden)
        w_out_final = session.run(r_w_out)

        total_time = end - start
    
    return (w_hidden_final, w_out_final), er, total_time
