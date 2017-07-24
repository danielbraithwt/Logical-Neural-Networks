import sys
sys.path.append('../lib/')

from RealSpaceLNFNetwork import train_cnf_network, train_dnf_network
from NeuralNetwork import train_perceptron_network, train_perceptron_network_general
from BooleanFormula import build_cnf, build_dnf, test_cnf
import numpy as np
import scipy.stats as ss
import random

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

def generateExpressions(n, num):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), 2**len(inputs))

    return np.array(list(map(lambda x: (inputs, x), outputs)))


if __name__ == '__main__':
    n = 2
    np.random.seed(1234)
    random.seed(1234)
    expressions = generateExpressions(n, 2**n)

    for e in expressions:
        data = e[0]
        targets = e[1]

        print()
        r_net_cnf, _, er_cnf, _ = train_cnf_network(n, data, targets, 100000)
        print("CNF Trained: ", er_cnf)
        r_net_dnf, _, er_dnf, _ = train_dnf_network(n, data, targets, 100000)
        print("DNF Trained: ", er_dnf)

        cnf = build_cnf(n, r_net_cnf)
        dnf = build_dnf(n, r_net_dnf)

        print()
        print("CNF: ", cnf)
        print()
        print("DNF: ", dnf)

        print()
        print("CNF Wrong: ", test_cnf(cnf, data, targets))
        print("DNF Wrong: ", test_cnf(dnf, data, targets))

        
        
        
