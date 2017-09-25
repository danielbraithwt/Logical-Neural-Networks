import sys
sys.path.append('../../lib/')

import MultiOutLNN
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sigmoid(inputs, weights, bias):
    z = tf.add(tf.matmul(inputs, tf.transpose(weights)), bias)
    return tf.nn.sigmoid(z)

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


def linear(inputs, weights, bias):
    return tf.add(tf.matmul(inputs, tf.transpose(weights)), bias)

def transform_weights(weights):
    return tf.log((1 + tf.exp(-weights)))

def inv_transform(weights):
    return -np.log(np.exp(weights)-1)

def transform(weights):
    return np.log(1 + np.exp(-weights))


def gen_weights(shape):
    var = np.sqrt(np.log((3 * shape[1] + 4)/4))
    mean = np.log(16.0/(shape[1]**2 * (3 * shape[1] + 4)))/2
    #var = np.sqrt(np.log(4 * (4 + 3*shape[1])))
    #mean = -(1.0/2.0) * np.log(shape[1]**2 * (4 + 3*shape[1]))
    initial = np.random.lognormal(mean, var, shape)
    w = inv_transform(initial)
    #print(transform(w))

    #w = np.random.normal(0, 1.0/shape[1], shape)

    return w

def construct_network(num_inputs, hidden_layers, num_outputs, addNot):
    network = []
    
    layers = hidden_layers
    layers.append(num_outputs)

    layer_ins = num_inputs
    for idx in range(len(layers)):
        l = layers[idx]
        weights = None
        if addNot:
            weights = tf.Variable(gen_weights((l, 2 * layer_ins)), dtype='float32')
        else:
            weights = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')
        bias = tf.Variable(np.zeros((1, l)), dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network

def apply_network(network, activations, iput, addNot):
    prev_out = iput
    for idx in range(len(network)):
        if addNot:
            prev_out = tf.concat([prev_out, 1 - prev_out], axis=1)
        
        layer = network[idx]
        act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        out = act(prev_out, w, b)
        prev_out = out

    return prev_out

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + "/")
print(tf.gfile.Exists(dir_path + "/"))

mnist = input_data.read_data_sets(dir_path + "/", one_hot=False)

X_train = mnist.train.images
#Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print(len(X_train[0]))

iterations = 30
num_inputs = len(X_train[0])
encoded_size = 10

activations = [noisy_or_activation, noisy_and_activation, noisy_and_activation]#
encoder_net = construct_network(num_inputs, [60, 30], encoded_size, False)
decoder_net = construct_network(encoded_size, [30, 60], num_inputs, False)

examples = tf.constant(X_train)

random_example = tf.train.slice_input_producer([examples],
                                                           shuffle=False)

example_batch = tf.train.batch([random_example],
                                          batch_size=1)

weights = []
for layer in encoder_net:
    weights.append(1 - tf.nn.sigmoid(layer[0]))

for layer in decoder_net:
    weights.append(1 - tf.nn.sigmoid(layer[0]))


l1_reg = tf.contrib.layers.l1_regularizer(0.0001)
regularizer = tf.contrib.layers.apply_regularization(l1_reg, weights)

x = tf.placeholder("float32", )

encoded = apply_network(encoder_net, activations, x, False)
decoded = apply_network(decoder_net, np.flip(activations, 0), encoded, False)

errors = tf.pow(x - decoded, 2)
error = tf.reduce_sum(errors)

train_op = tf.train.AdamOptimizer(0.001).minimize(error + regularizer)
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    #print(session.run(error, feed_dict={x:[X_train[0]]}))
    for i in range(iterations):
        for ex in X_train:
            e, _ = session.run([error, train_op], feed_dict={x:[ex]})
            #print(e)

        er = 0   
        for d in range(len(X_train)):
            er += session.run(error, feed_dict={x:[X_train[d]]})

        print(session.run(regularizer))
        print(er/len(X_train))

    enc = []
    for l in encoder_net:
        weights, _ = session.run(l)
        enc.append(weights)

    dec = []
    for l in decoder_net:
        weights, _ = session.run(l)
        dec.append(weights)

    np.save('mnist-autoencoder', [enc, dec])

    er = 0   
    for d in range(len(X_train)):
        er += session.run(error, feed_dict={x:[X_train[d]]})

    print(er/len(X_train))

    er = 0   
    for d in range(len(X_test)):
        er += session.run(error, feed_dict={x:[X_test[d]]})

    print(er/len(X_test))

    #colors = cm.rainbow(np.linspace(0, 1, 10))
    #compressed = [[], [], [], [], [], [], [], [], [], []]
    #for j in range(len(X_test)):
    #    c = session.run(encoded, feed_dict={x:[X_test[d]]})[0]
        #print(c)
        #print(Y_test[j])
    #    compressed[int(Y_test[j])].append(c)

    #for j in range(10):
    #    color = colors[j]
    #    d = np.array(compressed[j])
    #    x_plot = d[:,0]
    #    y_plot = d[:,1]
    #    plt.scatter(x_plot, y_plot, color=color)

    #plt.show()
    


            
##    coord = tf.train.Coordinator()
##    threads = tf.train.start_queue_runners(sess=session,coord=coord)
##
##    for i in range(iterations):
##        batch_ex = session.run([example_batch])
##        session.run([train_op], feed_dict={x:batch_ex})
##
##        if i % len(data) == 0:
##            er = 0   
##            for d in range(len(data)):
##                er += session.run(error, feed_dict={x:[data[d]]})
##
##            print(er)
##
##    coord.request_stop()
##    coord.join(threads)
