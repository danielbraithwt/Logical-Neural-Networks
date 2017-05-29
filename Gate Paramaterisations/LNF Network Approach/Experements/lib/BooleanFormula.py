from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network, train_perceptron_general_network
import numpy as np
import Pruning
import random

class Atom():
    def __init__(self, name):
        self.name = name

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

    def __repr__(self):
        s = ""

        for i in range(len(self.literals)):
            s += "(" + str(self.literals[i]) + ")"
            if not i == len(self.literals)-1:
                s += " OR "

        return s


def build_cnf(n, network):
    pruned_network = Pruning.relevance_pruning(network, 1.0)

    hidden_w = pruned_network[0]
    out_w = pruned_network[1]

    present = [i for i in range(1,len(out_w)) if out_w[i] > 0]
    raw_disjunctions = [hidden_w[i-1] for i in present]

    atoms = []
    for i in range(n):
        atoms.append(Atom("{}".format(i)))
        atoms.append(Atom("NOT {}".format(i)))

    disjunctions = []
    for weights in raw_disjunctions:
        mask = [w for w in range(1, len(weights)) if weights[w] > 0]
        a = [atoms[i-1] for i in mask]
        disjunctions.append(Or(a))


    return And(disjunctions)
    

def get_inputs(row):
    atoms = {}
    for i in range(len(row)):
        atoms["{}".format(i)] = row[i] == 1
        atoms["NOT {}".format(i)] = (1 - row[i]) == 1

    return atoms

def test_cnf(cnf, data, targets):
    wrong = 0

    for i in range(len(data)):
        row = data[i]
        inputs = get_inputs(row)
        t_hat = cnf.apply(inputs)

        if not t_hat == targets[i]:
            wrong += 1

    return wrong


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


#plt.switch_backend("TkAgg")  
n = 3
if __name__ == '__main__':
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    network, loss, time = train_cnf_network(n, data, targets)
    cnf = build_cnf(n, network)

    print(data)
    print(targets)

    print("CNF:")
    print(cnf)
    print()
    print("Testing")
    wrong = test_cnf(cnf, data, targets)
    print("{} wrong".format(wrong))
    
