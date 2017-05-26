import sys
sys.path.append('../lib/')
from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network, train_perceptron_network_general, run_cnf_network, run_dnf_network, run_perceptron_network, train_perceptron_general_network, run_perceptron_general_network
from NeuralNetwork import train_network, noisy_or_activation, noisy_and_activation, train_perceptron_network_general
import numpy as np
import random

#data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
#targets = [1.0, 1.0, 1.0, 0.0]

#data = [
#    [0.0, 0.0],
#    [1.0, 0.0],
#    [0.0, 1.0],
#    [1.0, 1.0],
#]

#targets = [1.0, 0.0, 0.0, 1.0]

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

e = generateExpressions(6)[0]
print("CNF: ", train_cnf_network(6, e[0].tolist(), e[1].tolist())[1])
print("DNF: ", train_dnf_network(6, e[0].tolist(), e[1].tolist())[1])
print("PCEP: ", train_perceptron_network(6, e[0].tolist(), e[1].tolist())[1])
print("PCEPG: ", train_perceptron_general_network(6, e[0].tolist(), e[1].tolist())[1])

