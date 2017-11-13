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
    

class AtomContinous():
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.get_name()

    def apply(self, vals):
        return vals[int(self.name)]

    def __repr__(self):
        return self.name

class AndContinous():
    def __init__(self, literals):
        self.literals = literals

    def apply(self, vals):
        res = 1
        for l in self.literals:
            res *= np.power(0.0001, 1 - l.apply(vals))

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
    
class OrContinous():
    def __init__(self, literals):
        self.literals = literals

    def apply(self, vals):
        res = 1
        for l in self.literals:
            res *= np.power(0.0001, l.apply(vals))

        return 1 - res

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

class NotContinous():
    def __init__(self, literal):
        self.literal = literal

    def apply(self, vals):
        return 1 -  self.literal.apply(vals)

    def get_literals(self):
        return self.literals

    def __repr__(self):
        return "NOT (" + str(self.literal) + ")"


def sigmoid(z):
    return 1/(1 + np.exp(-z))

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
        #weights = tf.Variable(np.random.uniform(-1.0, 1.0, (l, 2 * layer_ins)), dtype='float32')
        #bias = tf.Variable(np.random.uniform(-1.0, 1.0, (1, l)), dtype='float32')
        weights = None
        if addNot:
            weights = tf.Variable(gen_weights((l, 2 * layer_ins)), dtype='float32')
        else:
            weights = tf.Variable(gen_weights((l, layer_ins)), dtype='float32')
        bias = tf.Variable(np.zeros((1, l)) + 27.6309, dtype='float32')

        network.append([weights, bias])
        layer_ins = l

    return network

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def train_lnn(data, targets, iterations, num_inputs, hidden_layers, num_outputs, activations, addNot=True):
    #data = list(map(lambda x: transform_input(x), data))
    network = construct_network(num_inputs, hidden_layers, num_outputs, addNot)

    examples = tf.constant(data)
    labels = tf.constant(targets.tolist())

    random_example, random_label = tf.train.slice_input_producer([examples, labels],
                                                           shuffle=False)

    example_batch, label_batch = tf.train.batch([random_example, random_label],
                                          batch_size=25)

    x = tf.placeholder("float32", )
    y = tf.placeholder("float32", )

    prev_out = x
    for idx in range(len(network)):
        if addNot:
            prev_out = tf.concat([prev_out, 1 - prev_out], axis=1)
        
        layer = network[idx]
        act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        out = act(prev_out, w, b)
        prev_out = out

    y_hat = prev_out
    y_hat = y_hat * (1/tf.reduce_sum(y_hat))

    y_hat_prime = y_hat
    y_hat_prime_0 = tf.clip_by_value(y_hat_prime, 1e-20, 1)
    y_hat_prime_1 = tf.clip_by_value(1 - y_hat_prime, 1e-20, 1)
    errors = -(y * tf.log(y_hat_prime_0) + (1-y) * tf.log(y_hat_prime_1))#
    error = tf.reduce_sum(errors)

    #errors = tf.pow(y - y_hat, 2)
    #error = tf.reduce_sum(errors)

    #error = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=[10 * y_hat])

    minimize = error #+ big_weights

    #train_op = tf.train.AdamOptimizer(0.01).minimize(error)
    train_op = tf.train.AdamOptimizer(0.001).minimize(error)
    model = tf.global_variables_initializer()

    savable = []
    for l in network:
        savable.append(l[0])
        savable.append(l[1])
        
    #saver = tf.train.Saver(savable)

    with tf.Session() as session:
        session.run(model)
        #saver.restore(session, 'model.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,coord=coord)

        for i in range(iterations):
        #for i in range(166352):
        #for i in range(0):
            batch_ex, batch_l = session.run([example_batch, label_batch])
            session.run([train_op], feed_dict={x:batch_ex, y:batch_l})
            #print(session.run([y_hat], feed_dict={x:batch_ex, y:batch_l}))
            #print(session.run(y, feed_dict={y:batch_l}))
    
            if i % len(data) == 0:
                #session.run(train_op_big_weights)
                er = 0

                
                for d in range(len(data)):
                    
                    er += session.run(error, feed_dict={x:[data[d]], y:targets[d]})
                print()
                print(i)
                print(er)
##                print(session.run(y_hat, feed_dict={x:[data[1]], y:targets[1]}))
##                print(targets[1])
##                print(session.run(error, feed_dict={x:[data[1]], y:[targets[1]]}))
##                print()
##                print(session.run(y_hat, feed_dict={x:[data[10]], y:targets[10]}))
##                print(targets[10])
##                print(session.run(error, feed_dict={x:[data[10]], y:[targets[10]]}))
##                print()
##                print(session.run(y_hat, feed_dict={x:[data[50]], y:targets[50]}))
##                print(targets[50])
##                print(session.run(error, feed_dict={x:[data[50]], y:[targets[50]]}))

                #if er < 0.0000000001:
                #    break
        #saver.save(session, 'model.ckpt')


        final_network = []
        for layer in network:
            weights, bias = session.run(layer)
            #print(sigmoid(weights))
            final_network.append([weights, bias])

        np.save('network', final_network)

        coord.request_stop()
        coord.join(threads)

    return final_network


def run_lnn(data, targets, network, activations, addNot=True):
    x = tf.placeholder("float32", )
    y = tf.placeholder("float32", )

    prev_out = x
    tmp = None
    for idx in range(len(network)):
        tmp = prev_out
        if addNot:
            prev_out = tf.concat([prev_out, 1 - prev_out], axis=1)
        
        layer = network[idx]
        act = activations[idx]
        
        w = layer[0]
        b = layer[1]

        out = act(prev_out, w, b)
        prev_out = out

    y_hat = prev_out
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

            #print(pred)
            #print(targets[i])
            predicted_prob = 0
            predicted = -1

            if not np.round(pred)[0] == targets[i][0]:
                wrong += 1
##            for j in range(len(pred)):
##                if pred[j] > predicted_prob:
##                    predicted = j
##                    predicted_prob = pred[j]
##
##                if targets[i][j] == 1:
##                    actual = j

##            print(predicted)
##            print(actual)
##            print()
##
##            if not (actual == predicted):
##                wrong += 1

        return wrong
            


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

def test_fuzzy_rules(cnf, data, targets):
    wrong = 0

    for i in range(len(data)):
        row = data[i]
        inputs = row#get_inputs(row)

        pred = -1
        pred_val = -1
        actual = -1

        for j in range(len(cnf)):
            t_hat = cnf[j].apply(inputs)
            #print(t_hat)
            if t_hat > pred_val:
                pred_val = t_hat
                pred = j

            if targets[i][j] == 1:
                actual = j

        #print(pred, " : ", actual)

        if not pred == actual:
            wrong += 1

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
        #print()
        num_expr = len(expressions)
        for i in range(num_expr):
            expressions = np.append(expressions, Not(expressions[i]))
        print(expressions)
        
        t = types[idx]
        l = net[idx]
        w = l[0]

        formulas = []
        for neuron in w:
            neuron = np.round(sigmoid(neuron))
            print(neuron)
            considered = expressions[neuron == 0]
            print(considered)

            #if len(considered) == 0:
                #continue
            
            if t == "AND":
                formulas.append(And(considered))
            else:
                formulas.append(Or(considered))

        expressions = np.array(formulas)

    return expressions


def ExtractFuzzyRules(n, net, types, threshold, dp, nots=True):
    atoms = []
    for i in range(n):
        atoms.append(AtomContinous("{}".format(i)))
    atoms = np.array(atoms)


    expressions = atoms

    for idx in range(len(net)):
        #print()
        if nots:
            num_expr = len(expressions)
            for i in range(num_expr):
                expressions = np.append(expressions, NotContinous(expressions[i]))
        #print(expressions)
        
        t = types[idx]
        l = net[idx]
        w = l[0]

        formulas = []
        for neuron in w:
            neuron = np.round(sigmoid(neuron), dp)
            #print(neuron)
            considered = expressions[neuron <= threshold]
            #print(considered)

            #if len(considered) == 0:
                #continue
            
            if t == "AND":
                formulas.append(AndContinous(considered))
            else:
                formulas.append(OrContinous(considered))

        expressions = np.array(formulas)

    return expressions




if __name__ == "__main__":

    np.random.seed(1234)
    random.seed(1234)

    N = 7
    expression = generateExpressions(N)[0]
    data = expression[0]
    targets = expression[1]
    #200000
    res = train_lnn(data, targets, 800000, N, [128], 1, [noisy_or_activation, noisy_and_activation], True)
    for layer in res:
        print(layer)
        print()

    #hidden_weights = res[1][0][0]
    #hidden_bias = res[1][0][1]
    #output_weights = res[1][1][0]
    #output_bias = res[1][1][1]

    rule = ExtractRules(N, res, ["OR", "AND"])
    print(len(rule))
    rule = rule[0]
    print(rule)

    print(data)
    print(targets)

    print(test(rule, data, targets))


        
