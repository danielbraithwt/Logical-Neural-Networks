from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + "/")
print(tf.gfile.Exists(dir_path + "/"))

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

def gen_weights(shape):
    eps = 0.99 * np.clip(1e-10, 0.9999999999, np.abs(np.random.normal(0, 1, shape)))
    return np.log(-(eps/(eps-1)))

def construct_network(num_inputs, hidden_layers, num_outputs):
    network = []
    
    layers = hidden_layers
    layers.append(num_outputs)

    layer_ins = num_inputs
    for l in layers:
        #weights = tf.Variable(1 * np.zeros((l, 2 * layer_ins)), dtype='float32')
        #bias = tf.Variable(1 * np.zeros((1, l)), dtype='float32')

        weights = tf.Variable(gen_weights((l, 2 * layer_ins)), dtype='float32')
        bias = tf.Variable(gen_weights((1, l)), dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network

mnist = input_data.read_data_sets(dir_path + "/", one_hot=True)

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print(len(X_train[0]))

num_inputs = len(X_train[0])
num_outputs = len(Y_train[0])

## CONFIGURATION ##
batch_size = 1#len(X_train)
hidden_layers = [30]
activations = [noisy_or_activation, noisy_and_activation]
iterations = len(X_train) * 20
## ## ## ## ## ## ##

network = construct_network(num_inputs, hidden_layers, num_outputs)

x = tf.placeholder("float32", )
y = tf.placeholder("float32", )

prev_out = x
for idx in range(len(network)):
    prev_out = tf.concat([prev_out, 1 - prev_out], axis=1)
    
    layer = network[idx]
    act = activations[idx]
    
    w = layer[0]
    b = layer[1]

    out = act(prev_out, w, b)
    prev_out = out

y_hat = prev_out#tf.nn.softmax(prev_out)

#big_weights = 0.0
#for layer in network:
#    w = layer[0]
#    big_weights = tf.add(big_weights, tf.reduce_max(tf.abs(w)))
#big_weights = 0.0000005 * (-tf.log(big_weights))

#y_hat_prime = tf.reduce_sum(y_hat)
#y_hat_prime_0 = tf.clip_by_value(y_hat_prime, 1e-16, 1)
#y_hat_prime_1 = tf.clip_by_value(1 - y_hat_prime, 1e-16, 1)
#errors = tf.pow(y - y_hat, 2)#y * tf.log(y_hat_prime_0) + (1-y) * tf.log(y_hat_prime_1)#
#error = tf.reduce_sum(errors)
sm = tf.nn.softmax(y_hat)
errors = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
error = tf.reduce_sum(errors)

#minimize = error #+ big_weights

#train_op = tf.train.AdamOptimizer(0.01).minimize(error)
train_op = tf.train.AdamOptimizer(0.1).minimize(error)
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    #batch_ex, batch_l = mnist.train.next_batch(1)
    #print(session.run(y_hat, feed_dict={x:batch_ex, y:batch_l}))
    #print(session.run(sm, feed_dict={x:batch_ex, y:batch_l}))
    #print(FADW)
##    print(session.run(errors, feed_dict={x:batch_ex, y:batch_l}))
##    print(session.run(error, feed_dict={x:batch_ex, y:batch_l}))
##
##    for i in range(10000):
##        session.run(train_op, feed_dict={x:batch_ex, y:batch_l})
##    
##
##    print(session.run(y_hat, feed_dict={x:batch_ex, y:batch_l}))
##    print(session.run(errors, feed_dict={x:batch_ex, y:batch_l}))
##    print(session.run(error, feed_dict={x:batch_ex, y:batch_l}))

    for i in range(iterations):
        #print(i)
        batch_ex, batch_l = mnist.train.next_batch(batch_size)
        session.run([train_op], feed_dict={x:batch_ex, y:batch_l})

        if i % 30 == 0:
            er = session.run(error, feed_dict={x:X_train, y:Y_train})
            er_test = session.run(error, feed_dict={x:X_test, y:Y_test})
            print()
            print(session.run(y_hat, feed_dict={x:[X_train[0]], y:Y_train[0]}))
            print(session.run(sm, feed_dict={x:[X_train[0]], y:Y_train[0]}))
            print(Y_train[0])
            print(i)
            print(er)
            print(er_test)

        #if i % len(X_train) * 10 == 0:
        #    er = session.run(error, feed_dict={x:X_train, y:Y_train})
        #    print()
        #    print(i)
        #    print(er)
            #print(reg)

            #if er < 0.0000000001:
            #    break

    final_network = []
    for layer in network:
        weights, bias = session.run(layer)
        print(sigmoid(weights))
        final_network.append([np.round(sigmoid(weights)), np.round(sigmoid(bias))])
