import sys
sys.path.append('../lib/')

from Pruning import relevance_pruning
from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network, run_cnf_network, run_dnf_network
import numpy as np
import random


def print_network(net):
    hidden = net[0]
    output = net[1]

    print("Hidden")
    for n in hidden:
        print(n)

    print("\nOutput")
    print(output)

#data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
#targets = np.array([0.0, 1.0, 1.0, 0.0])

#data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
#targets = np.array([1.0, 1.0, 0.0, 1.0])

data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
targets = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

network, error, time = train_dnf_network(3, data, targets)
print("Not Pruned")
print(print_network(network))
print("Error: " + str(error))

pruned_net = relevance_pruning(network, 1.0)
pruned_error = run_dnf_network(3, pruned_net, data, targets)

print("\n\nPruned")
print(print_network(pruned_net))
print("Error: " + str(pruned_error))
