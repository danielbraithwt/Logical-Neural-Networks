import sys
sys.path.append('../lib/')
from NeuralNetwork import train_network_for_loss, noisy_or_activation, noisy_and_activation, train_perceptron_network_general
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss

plt.switch_backend("TkAgg")

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


N = 10
name = "CNF"

lr = 0.0001

ite = 30001
interval = 100
e = generateExpressions(N)[0]
data = e[0]
targets = e[1]

losses = train_network_for_loss(N, data, targets, noisy_or_activation, noisy_and_activation, 0, ite, lr, interval)

x_axis = np.array(range(0, ite, interval))

plt.plot(x_axis, losses, '-o', color='b', label='CNF')

plt.ylabel("Accuracy")
plt.xlabel("Training Iterations")
plt.xlim([0, ite])
plt.legend(loc='best')
plt.show()
#plt.savefig("{}-(N={},lr={})-{}-iterations.png".format(name, str(N), str(lr), str(ite)))
#plt.clf()
