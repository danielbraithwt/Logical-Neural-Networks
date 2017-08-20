import sys
import time
import tensorflow as tf
import random
import numpy as np


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

class Atom():
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.get_name()

    def apply(self, vals):
        return (self.name in vals)

    def __repr__(self):
        return self.name
    

class And():
    def __init__(self, literals):
        self.literals = literals

    def apply(self, vals):
        res = True
        for l in self.literals:
            res = res and l.apply(vals)

        return res

    def get_literals(self):
        return self.literals

    def __repr__(self):
        s = ""

        for i in range(len(self.literals)):
            s += "(" + str(self.literals[i]) + ")"
            if not i == len(self.literals)-1:
                s += " AND "

        return s
    
class Or():
    def __init__(self, literals):
        self.literals = literals

    def apply(self, vals):
        res = False
        for l in self.literals:
            res = res or l.apply(vals)

        return res

    def get_literals(self):
        return self.literals

    def __eq__(self, other):
        l2 = other.get_literals()

        for atm in l2:
            if not atm in self.literals:
                return False

        return True

    def __repr__(self):
        s = ""

        for i in range(len(self.literals)):
            s += "(" + str(self.literals[i]) + ")"
            if not i == len(self.literals)-1:
                s += " OR "

        return s

class Not():
    def __init__(self, literal):
        self.literal = literal

    def apply(self, vals):
        return not self.literal.apply(vals)

    def get_literals(self):
        return self.literals

    def __repr__(self):
        return "NOT (" + str(self.literal) + ")"


def generateExpressions(n):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), 1)

    return np.array(list(map(lambda x: (inputs, x), outputs)))

##def noisy_or_activation(inputs, weights, bias):
##    t_w = transform_weights(weights)
##    t_b = transform_weights(bias)
##    
##    z = tf.add(tf.reduce_sum(tf.multiply(inputs, t_w), axis=1), t_b)
##    return 1 - tf.exp(-z)
##
##def noisy_and_activation(inputs, weights, bias):
##    t_w = transform_weights(weights)
##    t_b = transform_weights(bias)
##
##    z = tf.add(tf.reduce_sum(tf.multiply(1 - inputs, t_w), axis=1), t_b)
##    return tf.exp(-z)

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
        #transformed.append(1-i)
    return transformed

def sigmoid(weights):
    return 1.0/(1 + tf.exp(-weights))
    

def _train_network(N, data, targets, iterations, hidden_activation, output_activation):
    data = list(map(lambda x: transform_input(x), data))

    examples = tf.constant(data)
    labels = tf.constant(targets.tolist())

    random_example, random_label = tf.train.slice_input_producer([examples, labels],
                                                           shuffle=False)

    example_batch, label_batch = tf.train.batch([random_example, random_label],
                                          batch_size=1)
    
    # Data and target variables
    x = tf.placeholder("float64", [None, None])
    y = tf.placeholder("float64", )


    w_hidden = tf.Variable(np.random.normal(0.0, 1.0, (N, N)), dtype='float64')
    g_hidden = tf.Variable(np.random.normal(0.0, 1.0, (N, 1)), dtype='float64')
    b_hidden = tf.Variable(np.transpose(np.random.normal(0.0, 1.0, (N, 1))), dtype='float64')

    w_out = tf.Variable(np.random.normal(0.0, 1.0, (1, N)), dtype='float64')
    g_out = tf.Variable(np.random.normal(0.0, 1.0, (1)), dtype='float64')
    b_out = tf.Variable(np.random.normal(0.0, 1.0, (1)), dtype='float64')


    g_h = sigmoid(g_hidden)
    g_o = sigmoid(g_out)

    hidden_out = hidden_activation(x, w_hidden, b_hidden)
    hidden_out = g_h * hidden_out + (1-g_h) * (1-hidden_out)
    y_hat = output_activation(hidden_out, w_out, b_out)
    y_hat = g_o * y_hat + (1-g_o) * (1-y_hat)

    errors = -(y * tf.log(y_hat) + (1-y) * tf.log(1-y_hat))#tf.pow(y - y_hat, 2)
    error = tf.reduce_sum(errors)

    train_op_error = tf.train.AdamOptimizer(0.005).minimize(error)
    #train_op_gate = tf.train.AdamOptimizer(0.5).minimize(error, var_list=[g_hidden, g_out])

    #t_w_hidden = sigmoid(w_hidden)
    #t_w_out = sigmoid(w_out)

    #r_w_hidden = tf.round(t_w_hidden)
    #r_w_out = tf.round(t_w_out)

    model = tf.global_variables_initializer()
    start = time.time()

    with tf.Session() as session:
        session.run(model)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):      
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run([train_op_error], feed_dict={x:batch_ex, y:batch_l})

            if i % 10 == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})
                print(er)
                print(session.run(g_h))
            
        er = 0
        for d in range(len(data)):
            er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})

        end = time.time()

        coord.request_stop()
        coord.join(threads)
        
        r_hidden_final = session.run(r_w_hidden)
        r_out_final = session.run(r_w_out)

        total_time = end - start

        w_hidden_final = session.run(w_hidden)
        b_hidden_final = session.run(b_hidden)

        w_out_final = session.run(w_out)
        b_out_final = session.run(b_out)
    
    return (r_hidden_final, r_out_final), ((w_hidden_final, b_hidden_final), (w_out_final, b_out_final)), er, total_time




np.random.seed(1234)
random.seed(1234)

N = 9
expression = generateExpressions(N)[0]
data = expression[0]
targets = expression[1]
#200000
res = _train_network(N, data, targets, 100000, noisy_and_activation, noisy_or_activation)
for layer in res:
    print(layer)
    print()

#hidden_weights = res[1][0][0]
#hidden_bias = res[1][0][1]
#output_weights = res[1][1][0]
#output_bias = res[1][1][1]


print(data)
print(targets)

rule = ExtractRules(N, res, ["AND","OR"])
print(len(rule))
rule = rule[0]
print(rule)

print(test(rule, data, targets))
