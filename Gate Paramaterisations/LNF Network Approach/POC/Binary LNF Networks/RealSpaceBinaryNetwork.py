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

def weight_bonus(weights):
    return tf.reduce_sum(tf.abs(weights))

def sigmoid(weights):
    return 1.0/(1 + tf.exp(-weights))

def train_network(N, data, targets, iterations, lr):
    BATCH_SIZE = 1
    data = list(map(lambda x: transform_input(x), data))

    examples = tf.constant(data)
    labels = tf.constant(targets.tolist())

    random_example, random_label = tf.train.slice_input_producer([examples, labels],
                                                           shuffle=False)

    example_batch, label_batch = tf.train.batch([random_example, random_label],
                                          batch_size=BATCH_SIZE)
    
    # Data and target variables
    x = tf.placeholder("float32", [None, None])
    y = tf.placeholder("float32", )


    w_hidden = tf.Variable(np.random.uniform(-1.0, 1.0, (2**N, 2*N)), dtype='float32')
    b_hidden = tf.Variable(np.transpose(np.random.uniform(-1.0, 1.0, (2**N, 1))), dtype='float32')

    w_out = tf.Variable(np.random.uniform(-1.0, 1.0, (1, 2**N)), dtype='float32')
    b_out = tf.Variable(np.random.uniform(-1.0, 1.0, (1)), dtype='float32')

    hidden_out = noisy_or_activation(x, w_hidden, b_hidden)
    y_hat = noisy_and_activation(hidden_out, w_out, b_out)

    errors = tf.pow(y - y_hat, 2)
    error = tf.reduce_sum(errors)
    bonus = -0.02 * (weight_bonus(w_hidden) + weight_bonus(w_out))

    train_op_error = tf.train.AdamOptimizer(0.01).minimize(error)
    train_op_bonus = tf.train.AdamOptimizer(0.0003).minimize(bonus)

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
            if i == len(data):
                session.run(train_op_bonus)

            if i % 100 == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})

                print(er)
                print(session.run(bonus))
                print()
            #print(batch_ex)
            #print(batch_l)
            #print()
        #print(session.run(b_hidden, feed_dict={x:data, y:targets}))
        #print(session.run(hidden_out, feed_dict={x:data, y:targets}))
        #print(session.run(train_op_error, feed_dict={x:data[0], y:targets[0]}))
        #print(session.run(error, feed_dict={x:data[0], y:targets[0]}))
##        for i in range(iterations):
##            for d in range(len(data)):
##                session.run(train_op_error, feed_dict={x:data[d], y:targets[d]})
##            session.run(train_op_bonus)
##
##            if i % 100 == 0:
##                er = 0
##                for d in range(len(data)):
##                    er += session.run(error, feed_dict={x:data[d], y:targets[d]})
##
##                print(er)
##                print(session.run(bonus))
##                print()

        #print(w_hidden)
        #print(w_out)

        end = time.time()

        coord.request_stop()
        coord.join(threads)
        
        w_hidden_final = session.run(r_w_hidden)
        w_out_final = session.run(r_w_out)
        print(end - start)
    
        return (w_hidden_final, w_out_final)

    

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

n = 5
np.random.seed(1234)
if __name__ == '__main__':
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    print(targets)

    r_net = train_network(n, data, targets, 40000, 0.003)
    print(r_net)
    #print(data)
    print(targets)
