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

def noisy_or_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)
    z = tf.add(tf.matmul(inputs, tf.transpose(t_w)), t_b)
    #z = tf.add(tf.reduce_sum(tf.multiply(t_w, inputs), axis=1), t_b)
    return 1 - tf.exp(-z)

def noisy_and_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)
    z = tf.add(tf.matmul(1 - inputs, tf.transpose(t_w)), t_b)
    #z = tf.add(tf.reduce_sum(tf.multiply(t_w, 1-inputs), axis=1), t_b)
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

def np_sigmoid(weights):
    return 1.0/(1 + np.exp(-weights))

#def gen_weights(shape):
#    eps = 0.99 * np.clip(1e-10, 0.9999999999, np.abs(np.random.normal(0, 1, shape)))
#    return np.log(-(eps/(eps-1)))


def create_network(num_inputs, hidden_layers, num_outputs):
    network = []
    layers = hidden_layers
    layers.append(num_outputs)

    layer_ins = num_inputs
    for l in layers:
        w = tf.Variable(np.random.uniform(-1.0, 1.0, (l, 2*layer_ins)), dtype='float32')
        wn = tf.Variable(np.random.uniform(-1.0, 1.0, (l, 2*layer_ins)), dtype='float32')
        #g = tf.Variable(np.random.uniform(-4.0, 4.0, (1, l)), dtype='float32')
        b = tf.Variable(np.random.uniform(-1.0, 1.0, (1, l)), dtype='float32')

        #w = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')
        #g = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')

        network.append([w, wn, b])
        layer_ins = l

    return network

def gen_weights(shape):
    eps = 0.5 * np.clip(1e-10, 0.9999999999, np.abs(np.random.normal(0, 1, shape)))
    return np.log(-(eps/(eps-1)))

def train_lnn(data, targets, iterations, num_inputs, hidden_layers, num_outputs, activations):
    network = create_network(num_inputs, hidden_layers, num_outputs)
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


    layer_input = x
    layer_input = tf.concat([layer_input, 1-layer_input], axis=1)
##    for i in range(len(network)):
##        layer = network[i]
##        w = layer[0]
##        g = sigmoid(layer[1])
##        b = layer[2]
##        act = activations[i]
##
##        gated_input = tf.add(g * layer_input, (1 - g) * (1 - layer_input))
##
##        layer_input = act(gated_input, w, b)
##        break;

    layer = network[0]
    w = layer[0]
    wn = layer[1]
    #g = sigmoid(layer[1])
    b = layer[2]
    act = activations[0]

    #out1 = act(layer_input, w, b)
    #out2 = act(1-layer_input, wn, b)

    out = act(layer_input, w, b)
    layer_input = out

    #gated_input = layer_input#tf.add(g * layer_input, (1 - g) * (1 - layer_input))

    layer_input = tf.concat([layer_input, 1-layer_input], axis=1)#out#act(gated_input, w, b)

    layer = network[1]
    w = layer[0]
    wn = layer[1]
    #g = sigmoid(layer[1])
    b = layer[2]
    act = activations[1]

    #out1 = act(layer_input, w, b)
    #out2 = act(1-layer_input, wn, b)

    out = act(layer_input, w, b)#tf.concat([out1, out2], axis=1)#tf.add(g * out1, (1-g) * out2)

    #gated_input = layer_input#tf.add(g * layer_input, (1 - g) * (1 - layer_input))

    layer_input = out#act(gated_input, w, b)

    y_hat = tf.reduce_sum(layer_input)

    #penalty = 0.0
    #for l in network:
    #    penalty = tf.add(penalty, tf.exp(-15 * tf.reduce_sum(tf.pow((sigmoid(l[1]) - 0.5), 2))))

    y_prime = tf.reduce_sum(y)
    errors = tf.pow(y_prime - y_hat, 2)#-(y * tf.log(y_hat) + (1-y) * tf.log(1-y_hat))#
    error = tf.reduce_sum(errors)

    train_op_error = tf.train.AdamOptimizer(0.01).minimize(error)

    #t_w_hidden = sigmoid(w_hidden)
    #t_w_out = sigmoid(w_out)

    #r_w_hidden = tf.round(t_w_hidden)
    #r_w_out = tf.round(t_w_out)

    model = tf.global_variables_initializer()
    start = time.time()

    with tf.Session() as session:
        session.run(model)

##        print(session.run(g, feed_dict={x:[data[1]], y:targets[1]}))
##        print(session.run(out1, feed_dict={x:[data[1]], y:targets[1]}))
##        print(session.run(out2, feed_dict={x:[data[1]], y:targets[1]}))
##        print(session.run(out, feed_dict={x:[data[1]], y:targets[1]}))

        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):      
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run(train_op_error, feed_dict={x:batch_ex, y:batch_l})

            if i % 100 == 0:
                er = 0
                for d in range(len(data)):
                    #print(session.run(y, feed_dict={x:[data[d]], y:targets[d]}))
                    #print(session.run(y_hat, feed_dict={x:[data[d]], y:targets[d]}))
                    er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})
                print(er)
                #print(session.run(penalty, feed_dict={x:[data[0]], y:targets[0]}))
                #print(session.run(gated_input, feed_dict={x:[data[0]], y:targets[0]}))
                print()
                #_, gate, _ = session.run(network[0])
                #print(session.run(g, feed_dict={x:[data[0]], y:targets[0]}))
            
        er = 0
        for d in range(len(data)):
            print(session.run(y_hat, feed_dict={x:[data[d]], y:targets[d]}))
            er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})

        end = time.time()

        coord.request_stop()
        coord.join(threads)

        total_time = end - start
##
##        final_network = []
##        for layer in network:
##            weights, gate = session.run(layer)
##            #print(np_sigmoid(weights))
##            final_network.append([np.round(np_sigmoid(weights)), np.round(np_sigmoid(gate))])#, np.round(sigmoid(bias))])
##
##    
##    return final_network


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
    atoms = []
    for i in range(len(row)):
        if row[i] == 1:
            atoms.append("{}".format(i))

    return atoms

def ExtractRules(N, network, types):
    atoms = []
    for i in range(N):
        atoms.append(Atom("{}".format(i)))
    atoms = np.array(atoms)

    expressions = atoms

    for idx in range(len(network)):
        negated_expressions = np.array(list(map(lambda x: Not(x), expressions)))

        t = types[idx]
        l = network[idx]
        w = l[0]
        g = l[1]

        print(g)

        formulas = []
        for neuron_id in range(len(w)):
            inputs = []
            gate = g[neuron_id]
            for i in range(len(gate)):
                if gate[i] == 1:
                    inputs.append(expressions[i])
                else:
                    inputs.append(negated_expressions[i])

            considered = np.array(inputs)[w[neuron_id] == 0]

            if t == "AND":
                formulas.append(And(considered))
            else:
                formulas.append(Or(considered))

        expressions = np.array(formulas)
    return expressions
    


np.random.seed(1234)
random.seed(1234)

N = 70
expression = generateExpressions(N)[0]
data = expression[0]
targets = expression[1]
#200000
print("Training")
res = train_lnn(data, targets, 80000, N, [3], 1, [noisy_and_activation, noisy_or_activation])
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
