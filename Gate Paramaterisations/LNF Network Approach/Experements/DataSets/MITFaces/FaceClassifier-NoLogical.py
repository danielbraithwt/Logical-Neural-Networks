import sys
sys.path.append('../../lib/')

import LoadFaces
import tensorflow as tf
import os
import numpy as np

def gen_weights(shape):
    return np.random.normal(0, 1.0/shape[1], shape)

def construct_network(num_inputs, hidden_layers, num_outputs):
    network = []
    
    layers = hidden_layers
    layers.append(num_outputs)

    layer_ins = num_inputs
    for idx in range(len(layers)):
        l = layers[idx]
        
        weights = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')
        bias = tf.Variable(np.zeros((1, l)), dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network


def train(data, targets, iterations, num_inputs, hidden_layers, num_outputs):
    #data = list(map(lambda x: transform_input(x), data))
    network = construct_network(num_inputs, hidden_layers, num_outputs)

    examples = tf.constant(data)
    labels = tf.constant(targets.tolist())

    random_example, random_label = tf.train.slice_input_producer([examples, labels],
                                                           shuffle=False)

    example_batch, label_batch = tf.train.batch([random_example, random_label],
                                          batch_size=1)

    x = tf.placeholder("float32", )
    y = tf.placeholder("float32", )

    prev_out = x
    for idx in range(len(network)):
        
        layer = network[idx]
        
        w = layer[0]
        b = layer[1]

        
        out = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        if not idx == len(network)-1:
            out = tf.nn.sigmoid(out)
        prev_out = out

    y_hat = prev_out
    #errors = tf.pow(y - y_hat, 2)
    #error = tf.reduce_sum(errors)

    error = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prev_out)

    minimize = error #+ big_weights

    #train_op = tf.train.AdamOptimizer(0.01).minimize(error)
    train_op = tf.train.AdamOptimizer(0.0001).minimize(error)
    model = tf.global_variables_initializer()

    savable = []
    for l in network:
        savable.append(l[0])
        savable.append(l[1])
        
    saver = tf.train.Saver(savable)

    with tf.Session() as session:
        session.run(model)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run([train_op], feed_dict={x:batch_ex, y:batch_l})
    
            if i % len(data) == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})

                print()
                print(i)
                print(er)
                print(session.run(y_hat, feed_dict={x:[data[1]], y:targets[1]}))
                print(targets[1])
                print(session.run(error, feed_dict={x:[data[1]], y:[targets[1]]}))
                print()
                print(session.run(y_hat, feed_dict={x:[data[10]], y:targets[10]}))
                print(targets[10])
                print(session.run(error, feed_dict={x:[data[10]], y:[targets[10]]}))
                print()
                print(session.run(y_hat, feed_dict={x:[data[50]], y:targets[50]}))
                print(targets[50])
                print(session.run(error, feed_dict={x:[data[50]], y:[targets[50]]}))


        final_network = []
        for layer in network:
            weights, bias = session.run(layer)
            final_network.append([weights, bias])

        np.save('network', final_network)

        coord.request_stop()
        coord.join(threads)

    return final_network

training, test = LoadFaces.read_data()

X_train = np.array(training[0])
Y_train = np.array(training[1])

X_test = np.array(test[0])
Y_test = np.array(test[1])

num_inputs = len(X_train[0])
num_outputs = len(Y_train[0])


train(X_train, Y_train, len(X_train) * 30, num_inputs, [1000, 500, 30], num_outputs)
