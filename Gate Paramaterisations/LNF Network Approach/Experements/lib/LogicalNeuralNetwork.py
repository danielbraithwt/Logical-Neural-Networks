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

    def negate(self):
        if self.name.startswith("NOT"):
            a = self.name.split(' ')[1]
            return Atom(a)

        return Atom("NOT {}".format(self.name))

    def apply(self, vals):
        return vals[self.name]

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
        transformed.append(1-i)
    return transformed

def construct_network(num_inputs, hidden_layers, num_outputs):
    network = []
    
    layers = hidden_layers
    layers.append(num_outputs)

    layer_ins = num_inputs
    for l in layers:
        weights = tf.Variable(np.random.uniform(-1.0, 1.0, (l, layer_ins)), dtype='float32')
        bias = tf.Variable(np.random.uniform(-1.0, 1.0, (1, l)), dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def train_lnn(data, targets, iterations, num_inputs, hidden_layers, num_outputs, activations):
    data = list(map(lambda x: transform_input(x), data))
    network = construct_network(2 * num_inputs, hidden_layers, num_outputs)

    examples = tf.constant(data)
    labels = tf.constant(targets.tolist())

    random_example, random_label = tf.train.slice_input_producer([examples, labels],
                                                           shuffle=False)

    example_batch, label_batch = tf.train.batch([random_example, random_label],
                                          batch_size=1)

    x = tf.placeholder("float32", [None, None])
    y = tf.placeholder("float32", )

    prev_out = x
    for idx in range(len(network)):
        layer = network[idx]
        act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        prev_out = act(prev_out, w, b)

    y_hat = prev_out

    errors = tf.pow(y - y_hat, 2)#-(y * tf.log(y_hat) + (1-y) * tf.log(1 - y_hat))#
    error = tf.reduce_sum(errors)

    train_op = tf.train.AdamOptimizer(0.01).minimize(error)
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        #print(session.run(prev_out, feed_dict={x:[data[0]], y:targets[0]}))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run([train_op], feed_dict={x:batch_ex, y:batch_l})
    
            if i % 10 == 0:
                er = 0
                for d in range(len(data)):
                    er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})
                print(er)

        final_network = []
        for layer in network:
            weights, bias = session.run(layer)
            final_network.append([np.round(sigmoid(weights)), np.round(sigmoid(bias))])

        coord.request_stop()
        coord.join(threads)

    return final_network


def test(cnf, data, targets):
    wrong = 0

    for i in range(len(data)):
        row = data[i]
        inputs = get_inputs(row)
        t_hat = cnf.apply(inputs)

        if not t_hat == targets[i]:
            wrong += 1

    return wrong

def get_inputs(row):
    atoms = {}
    for i in range(len(row)):
        atoms["{}".format(i)] = row[i] == 1
        atoms["NOT {}".format(i)] = (1 - row[i]) == 1

    return atoms

def ExtractRules(n, net, types):
    atoms = []
    for i in range(n):
        atoms.append(Atom("{}".format(i)))
        atoms.append(Atom("NOT {}".format(i)))
    atoms = np.array(atoms)


    expressions = atoms

    for idx in range(len(net)):
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




np.random.seed(1234)
random.seed(1234)

N = 5
expression = generateExpressions(N)[0]
data = expression[0]
targets = expression[1]

res = train_lnn(data, targets, 200000, N, [N, N], 1, [noisy_and_activation, noisy_or_activation, noisy_and_activation])
print(res)

#hidden_weights = res[1][0][0]
#hidden_bias = res[1][0][1]
#output_weights = res[1][1][0]
#output_bias = res[1][1][1]

rule = ExtractRules(N, res, ["AND", "OR", "AND"])[0]
print(rule)

print(data)
print(targets)

print(test(rule, data, targets))


        
