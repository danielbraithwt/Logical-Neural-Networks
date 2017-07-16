#from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network, train_perceptron_general_network
import RealSpaceLNFNetwork
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import random
import os

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
    pruned_network = network#Pruning.relevance_pruning(network, 1.0)

    hidden_w = pruned_network[0]
    out_w = pruned_network[1][0]
    

    present = [i for i in range(0,len(out_w)) if out_w[i] == 0]
    raw_disjunctions = [hidden_w[i] for i in present]

    atoms = []
    for i in range(n):
        atoms.append(Atom("{}".format(i)))
        atoms.append(Atom("NOT {}".format(i)))

    disjunctions = []
    for weights in raw_disjunctions:
        mask = [w for w in range(0, len(weights)) if weights[w] == 0]
        a = [atoms[i] for i in mask]
        disjunctions.append(Or(a))


    return And(disjunctions)

def build_dnf(n, network):
    pruned_network = network#Pruning.relevance_pruning(network, 1.0)

    hidden_w = pruned_network[0]
    out_w = pruned_network[1][0]
    

    present = [i for i in range(0,len(out_w)) if out_w[i] == 0]
    raw_conjunctions = [hidden_w[i] for i in present]

    atoms = []
    for i in range(n):
        atoms.append(Atom("{}".format(i)))
        atoms.append(Atom("NOT {}".format(i)))

    conjunctions = []
    for weights in raw_conjunctions:
        mask = [w for w in range(0, len(weights)) if weights[w] == 0]
        a = [atoms[i] for i in mask]
        conjunctions.append(And(a))


    #return And(disjunctions)
    return Or(conjunctions)
    

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


def compute_data(data, targets, i):
    ids = np.random.choice(len(data), 2**n - i, replace=False)
        
    d = np.take(data, ids)
    t = np.take(targets, ids)

    testData = []
    testTargets = []
    for i in range(0, len(data)):
        if not i in ids:
            testData.append(data[i])
            testTargets.append(targets[i])

    return (d, t), (testData, testTargets)

#pltswitch_backend("TkAgg")  
n = 6
if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    expression = generateExpressions(n)[0]
    data = expression[0]
    targets = expression[1]

    wrong_avgs = []
    wrong_mins = []
    wrong_maxs = []
    
    for i in range(0, 2**n):
        wrong = []

        for j in range(5):
            training, testing = compute_data(data, targets, i)

            r_net, _, er, _ = RealSpaceLNFNetwork.train_cnf_network(n, training[0], training[1], 90000)
            cnf = build_dnf(n, r_net)
            #if i == 0:
            #    print()
            #    print(cnf)
                
            w = test_cnf(cnf, data, targets)
            wrong.append(w)
            #sum_wrong += wrong
            #sum_err += er

        wrong = np.array(wrong)
        print(wrong)
        wrong_avgs.append(wrong.mean())
        wrong_mins.append(wrong.min())
        wrong_maxs.append(wrong.max())

        #avg_wrong = float(sum_wrong)/3.0
        #avgs.append(avg_wrong)
        #avgs_err.append(sum_err/3.0)
        #print(avg_wrong)

    x_axis = np.array(range(0, 2**n))

    plt.plot(x_axis, wrong_avgs, '-o', color='b', label='CNF Avg Inaccuracy')
    plt.plot(x_axis, wrong_mins, '-o', color='g', label='CNF Min Inaccuracy')
    plt.plot(x_axis, wrong_maxs, '-o', color='r', label='CNF Max Inaccuracy')
    #plt.plot(x_axis, avgs_err, '-o', color='b', label='CNF Error')

    plt.ylabel("Num Wrong")
    plt.xlabel("Number Of Training Examples Removed")
    plt.legend(loc='best')
    plt.savefig("cnf-descrete-generalization.png")
    #plt.clf()


##if __name__ == '__main__':
##    expression = generateExpressions(n)[0]
##    data = expression[0]
##    targets = expression[1]
##
##    training, testing = compute_data(data, targets, 1)
##
##    r_net, _, _, _ = RealSpaceLNFNetwork.train_cnf_network(n, training[0], training[1], 70000)
##    print(r_net)
##    cnf = build_cnf(n, r_net)
##
##    print(data)
##    print(targets)
##
##    print()
##    print(training)
##
##    print("{} : {}".format(len(training[0]), len(data)))
##
##    print("CNF:")
##    print(cnf)
##    print()
##    print("Testing")
##    wrong = test_cnf(cnf, data, targets)
##    print("{} wrong".format(wrong))
    
