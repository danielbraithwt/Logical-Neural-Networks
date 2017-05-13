import numpy as np
from CNFNetwork import CNFNetwork
from DNFNetwork import DNFNetwork
from PerceptronNetwork import PerceptronNetwork

def __loss(network, data, targets):
    predictions = __predict(network, data)
    return np.sum(np.power(np.subtract(targets, predictions), 2.0))

def __predict(network, data):
    t = np.array([network.fowardprop(d) for d in data])
    return t

def __computeGrad(network, neuron, pterb, data, targets):
    gradient = np.zeros(len(neuron.getWeights()))
    for k in range(0, len(neuron.getWeights())):
        g = np.zeros(len(neuron.getWeights()))
        g[k] = -pterb

        oldSSE = __loss(network, data, targets)
        neuron.updateWeights(g)
        newSSE = __loss(network, data, targets)
        neuron.updateWeights(-g)

        gradient[k] = (newSSE - oldSSE)/pterb

    return gradient

def trainNetwork(t, data, targets, inputNodes, numC, it=10000, lr=0.1):
    if t == 'cnf':
        network = CNFNetwork(inputNodes, numC)
    elif t == 'dnf':
        network = DNFNetwork(inputNodes, numC)
    elif t == 'perceptron':
        network = PerceptronNetwork(inputNodes, numC)

    pterb = 0.0001

    for i in range(1, it):
        for d in network.getHidden():
            g = __computeGrad(network, d, pterb, data, targets)
            d.setGrad(g * lr)

        g = __computeGrad(network, network.getOutput(), pterb, data, targets)

        network.getOutput().updateWeights(g * lr)
        for d in network.getHidden():
            d.applyGrad()

    return network, __loss(network, data, targets)
