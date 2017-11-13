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


def softmax(weights):
    e = np.exp(weights)
    return e/(sum(e))

def inv_transform(weights):
    return -np.log(np.exp(weights)-1)

def transform(weights):
    return np.log(1 + np.exp(-weights))


def gen_weights(shape):
    w = np.random.normal(0, 1.0/np.sqrt(shape[1]), shape)

    return w

def construct_network(num_inputs, hidden_layers, num_outputs):
    network = []
    
    layers = np.copy(hidden_layers).tolist()
    layers.append(num_outputs)

    layer_ins = num_inputs
    for idx in range(len(layers)):
        l = layers[idx]
        weights = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')
        bias = tf.Variable(np.zeros((1, l)), dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def train_lnn(data, targets, iterations, num_inputs, hidden_layers, num_outputs, sm=True):
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
        #act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        out = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        if not idx == len(network)-1:
            out = tf.nn.sigmoid(out)
        prev_out = out

    y_hat = prev_out
    
    if sm:
        y_hat = tf.nn.softmax(y_hat)
        #error = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=[y_hat])
    else:
        y_hat = tf.nn.sigmoid(y_hat)

    y_hat_prime = y_hat
    y_hat_prime_0 = tf.clip_by_value(y_hat_prime, 1e-20, 1)
    y_hat_prime_1 = tf.clip_by_value(1 - y_hat_prime, 1e-20, 1)
    errors = -(y * tf.log(y_hat_prime_0) + (1-y) * tf.log(y_hat_prime_1))#
    error = tf.reduce_sum(errors)

    minimize = error

    train_op = tf.train.AdamOptimizer(0.0005).minimize(error)
    model = tf.global_variables_initializer()


    with tf.Session() as session:
        session.run(model)
        #saver.restore(session, 'model.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run([train_op], feed_dict={x:batch_ex, y:batch_l})
            #print(session.run(y_hat, feed_dict={x:batch_ex, y:batch_l}))
            #print(session.run(y, feed_dict={x:batch_ex, y:batch_l}))
            #print(session.run(errors, feed_dict={x:batch_ex, y:batch_l}))
            #print(session.run(y, feed_dict={y:batch_l}))

            if i % len(data) == 0:
                er = 0
                for j in range(len(data)):
                    er += session.run(error, feed_dict={x:[data[j]], y:[targets[j]]})
                #print(er)

##                print()
##                print(session.run(y_hat, feed_dict={x:[data[0]], y:[targets[0]]}))
##                print(session.run(y, feed_dict={x:[data[0]], y:[targets[0]]}))
##                print()
##                print(session.run(y_hat, feed_dict={x:[data[100]], y:[targets[100]]}))
##                print(session.run(y, feed_dict={x:[data[100]], y:[targets[100]]}))
##                print()
##                print(session.run(y_hat, feed_dict={x:[data[4432]], y:[targets[4432]]}))
##                print(session.run(y, feed_dict={x:[data[4432]], y:[targets[4432]]}))
##                print()
    
        #saver.save(session, 'model.ckpt')


        final_network = []
        for layer in network:
            weights, bias = session.run(layer)
            #print(sigmoid(weights))
            final_network.append([weights, bias])

        #np.save('network', final_network)

        coord.request_stop()
        coord.join(threads)
        session.close()
        del session

    return final_network


def run_lnn(data, targets, network, sm):
    x = tf.placeholder("float32", )
    y = tf.placeholder("float32", )

    prev_out = x
    for idx in range(len(network)):
        layer = network[idx]
        #act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        out = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        if not idx == len(network)-1:
            print("HELLO")
            out = tf.nn.sigmoid(out)
        prev_out = out

    y_hat = prev_out
    
    if sm:
        y_hat = tf.nn.softmax(y_hat)
        #error = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=[y_hat])
    else:
        y_hat = tf.nn.sigmoid(y_hat)

        
    model = tf.global_variables_initializer()

    wrong = 0
    with tf.Session() as session:
        for i in range(len(data)):
            session.run(model)
            #print(session.run(tmp, feed_dict={x:[data[i]], y:targets[i]}))
            pred = session.run(y_hat, feed_dict={x:[data[i]], y:[targets[i]]})[0]
            #print(pred)
            #print()

            actual = -1

            if not np.round(pred)[0] == targets[i][0]:
                wrong += 1

            #print(pred)
            #print(targets[i])
##            predicted_prob = 0
##            predicted = -1
##            for j in range(len(pred)):
##                if pred[j] > predicted_prob:
##                    predicted = j
##                    predicted_prob = pred[j]
##
##                if targets[i][j] == 1:
##                    actual = j
##
##            #print(predicted)
##            #print(actual)
##            #print()
##
##            if not (actual == predicted):
##                wrong += 1

        return wrong
        session.close()
        del session
            


def test(cnf, data, targets):
    wrong = 0

    for i in range(len(data)):
        row = data[i]
        inputs = get_inputs(row)

        res = np.zeros(len(targets[0]))
        for j in range(len(cnf)):
            t_hat = cnf[j].apply(inputs)
            res[j] = t_hat

        #print(res)

        
        for j in range(len(res)):
            if not (res[j] == targets[i][j]):
                wrong += 1
                break

    return wrong

def get_inputs(row):
    atoms = []
    for i in range(len(row)):
        if row[i] == 1:
            atoms.append("{}".format(i))

    return atoms

def ExtractRules(n, net, types):
    atoms = []
    for i in range(n):
        atoms.append(Atom("{}".format(i)))
    atoms = np.array(atoms)


    expressions = atoms

    for idx in range(len(net)):
        num_expr = len(expressions)
        for i in range(num_expr):
            expressions = np.append(expressions, Not(expressions[i]))
        #print(expressions)
        
        t = types[idx]
        l = net[idx]
        w = l[0]

        formulas = []
        for neuron in w:
            considered = expressions[neuron == 0]
            
            if t == "AND":
                formulas.append(And(considered))
            else:
                formulas.append(Or(considered))

        expressions = np.array(formulas)

    return expressions


        
