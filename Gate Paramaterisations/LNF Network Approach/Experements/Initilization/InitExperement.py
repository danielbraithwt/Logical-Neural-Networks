import tensorflow as tf
import random
import numpy as np
from scipy import stats

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


def noisy_or_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = 0#transform_weights(bias)

    z = tf.add(tf.matmul(inputs, tf.transpose(t_w)), t_b)
    return 1 - tf.exp(-z)

def noisy_and_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = 0#transform_weights(bias)

    z = tf.add(tf.matmul(1 - inputs, tf.transpose(t_w)), t_b)
    return tf.exp(-z)

def transform_weights(weights):
    return tf.log((1 + tf.exp(-weights)))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        #transformed.append(1-i)
    return transformed


def softmax(weights):
    e = np.exp(weights)
    return e/(sum(e))

def inv_transform(weights):
    return -np.log(np.exp(weights)-1)

def transform(weights):
    return np.log(1 + np.exp(-weights))

def gen_weights(shape):

    #initial = np.abs(np.random.normal(-np.log(1/2)/shape[1], -np.log(1/2)/shape[1], shape))
    var = np.sqrt(np.log(4 * (4 + 3*shape[1])))
    mean = -(1.0/2.0) * np.log(shape[1]**2 * (4 + 3*shape[1]))
    
    #initial = np.random.lognormal(mean, var, shape)
    #initial = np.random.poisson((2.0/shape[1]), shape) + 0.0000000001
    #w = inv_transform(initial)

    #initial = []
    #for i in range(shape[0]):
    #    z_mean = np.random.exponential(1)
    #    weights = np.random.uniform(0, 2*(z_mean/shape[1]), shape[1])
    #    initial.append(weights)

    initial = stats.betaprime.rvs((14.0/(3.0 * shape[1])), (10.0/3.0), size=shape)
    w = inv_transform(np.array(initial))
    print(w)
    
    
    return w
    
    #print(np.abs(np.random.normal(0.5, 0.25, shape)))
    #eps = np.clip(np.abs(np.random.normal(0.5, 0.25, shape)), 1e-10, 0.9999999999)
    #print(eps)
    #return np.log(-(eps/(eps-1)))

def construct_network(num_inputs, hidden_layers, num_outputs):
    network = []
    
    layers = hidden_layers
    layers.append(num_outputs)

    layer_ins = num_inputs
    for l in layers:
        #weights = tf.Variable(np.random.uniform(-1.0, 1.0, (l, 2 * layer_ins)), dtype='float32')
        #bias = tf.Variable(np.random.uniform(-1.0, 1.0, (1, l)), dtype='float32')

        weights = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')
        bias = tf.Variable(np.zeros((1, l)) + 27.6309, dtype='float32')
        #bias = tf.Variable(gen_weights((1, l)), dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network

def test_lnn(data, num_inputs, hidden_layers, num_outputs, activations):
    data = list(map(lambda x: transform_input(x), data))
    network = construct_network(num_inputs, hidden_layers, num_outputs)

    examples = tf.constant(data)

    x = tf.placeholder("float32", )

    prev_out = x
    for idx in range(len(network)):
        #prev_out = tf.concat([prev_out, 1 - prev_out], axis=1)
        
        layer = network[idx]
        act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        out = act(prev_out, w, b)
        prev_out = out

    y_hat = tf.reduce_sum(prev_out)
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        activations = []
        for i in range(len(data)):
            activations.append(session.run(y_hat, feed_dict={x:[data[i]]}))
            #print(activations[i])

    return np.array(activations)
        


#np.random.seed(1234)
#random.seed(1234)

N = 10
expression = generateExpressions(N)[0]
data = expression[0]

acti = test_lnn(data, N, [30], 1, [noisy_or_activation, noisy_and_activation])
print(acti)
print("RESULTS")
print(acti.mean())
print(acti.std())
print(acti.var())
