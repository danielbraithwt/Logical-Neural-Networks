import numpy as np
from NeuralNetwork import noisy_or_activation, noisy_and_activation, perceptron_activation, train_network, train_perceptron_network_general, run_network, run_perceptron_network_general

def train_cnf_network(n, data, targets, iterations=20000, lr=0.01):
    return train_network(n, data, targets, noisy_or_activation, noisy_and_activation, 0, iterations, lr)

def train_dnf_network(n, data, targets, iterations=20000, lr=0.01):
    return train_network(n, data, targets, noisy_and_activation, noisy_or_activation, 0, iterations, lr)

def train_perceptron_network(n, data, targets, iterations=20000, lr=0.1):
    return train_network(n, data, targets, perceptron_activation, perceptron_activation, -np.infty, iterations, lr)

def train_perceptron_general_network(n, data, targets, iterations=20000, lr=0.1):
    return train_perceptron_network_general(n, data, targets, iterations, lr)

def run_cnf_network(n, net, data, targets):
    return run_network(n, net, data, targets, noisy_or_activation, noisy_and_activation)

def run_dnf_network(n, net, data, targets):
    return run_network(n, net, data, targets, noisy_and_activation, noisy_or_activation)

def run_perceptron_network(n, net, data, targets):
    return run_network(n, net, data, targets, perceptron_activation, perceptron_activation)

def run_perceptron_general_network(n, net, data, targets):
    return run_perceptron_network_general(n, net, data, targets)
