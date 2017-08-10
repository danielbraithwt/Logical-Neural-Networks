import numpy as np
import random
import NeuralNetwork

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


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def transform_input(iput):
    transformed = []
    for i in iput:
        transformed.append(i)
        transformed.append(1-i)
    return transformed

def ExtractRules(n, network):
    atoms = []
    for i in range(n):
        atoms.append(Atom("{}".format(i)))
        atoms.append(Atom("NOT {}".format(i)))
    atoms = np.array(atoms)


    expressions = atoms

    for i in range(len(network)):
        print(i)
        layer = network[i]
        #print(layer)
        #print(expressions)
        print()
        weights = layer[0]
        bias = layer[1]

        size = len(weights[0])
        if i == 0:
            size = int(size/2)

        inputs = np.array(__perms(size))
        print(inputs)
        print()
        if i == 0:
            inputs = np.array(list(map(lambda x: transform_input(x), inputs)))
    
        formulas = []
        for idx in range(len(weights)):
            w = weights[idx]
            b = bias[idx]

            rp = []

            for iput in inputs:
                #print(iput)
                #print(w)
                #print(b)
                #print(np.dot(w, iput))
                z = np.dot(w, iput) + b
                if sigmoid(z) > 0.5:
                    rp.append(iput)

            ands = []
            for pattern in rp:
                ands.append(And(expressions[pattern == 1]))
            formulas.append(Or(ands))
        expressions = np.array(formulas)

    return expressions[0]
    
##    hidden_formulas = []
##    for idx in range(len(hidden_weights)):
##        weights = hidden_weights[idx]
##        bias = hidden_bias[0][idx]
##        rp = []
##
##        for iput in hidden_inputs:
##            z = np.dot(weights, iput) + bias
##            if sigmoid(z) > 0.5:
##                rp.append(iput)
##
##        ands = []
##        for pattern in rp:
##            #print(pattern)
##            #print(pattern == 1)
##            #print(atoms[pattern == 1])
##            ands.append(And(atoms[pattern == 1]))
##
##        hidden_formulas.append(Or(ands))
##
##    hidden_formulas = np.array(hidden_formulas)
##
##    output_rp = []
##    for iput in inputs:
##        z = np.dot(output_weights, iput) + output_bias
##        if sigmoid(z) > 0.5:
##            output_rp.append(iput)
##
##    ands = []
##    for pattern in output_rp:
##        ands.append(And(hidden_formulas[pattern == 1]))
##
##
##    return Or(ands)
            

def get_inputs(row):
    atoms = {}
    for i in range(len(row)):
        atoms["{}".format(i)] = row[i] == 1
        atoms["NOT {}".format(i)] = (1 - row[i]) == 1

    return atoms

def test(cnf, data, targets):
    wrong = 0

    for i in range(len(data)):
        row = data[i]
        inputs = get_inputs(row)
        t_hat = cnf.apply(inputs)

        if not t_hat == targets[i]:
            wrong += 1

    return wrong


N = 8
expression = generateExpressions(N)[0]
data = expression[0]
targets = expression[1]

res = NeuralNetwork.train_perceptron_network_general(N, data, targets, 200000, 1)

#hidden_weights = res[1][0][0]
#hidden_bias = res[1][0][1]
#output_weights = res[1][1][0]
#output_bias = res[1][1][1]

rule = ExtractRules(N, res[1])
#print(rule)

print(res[2])
print(data)
print(targets)

print(test(rule, data, targets))
