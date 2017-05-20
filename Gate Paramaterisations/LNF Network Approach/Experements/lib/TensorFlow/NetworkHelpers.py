from NeuralNetwork import noisy_or_activation, noisy_and_activation, perceptron_activation, train_network

def train_cnf_network(n, data, targets):
    return train_network(n, data, targets, noisy_or_activation, noisy_and_activation, 0)

def train_dnf_network(n, data, targets):
    return train_network(n, data, targets, noisy_and_activation, noisy_or_activation, 0)

def train_perceptron_network(n, data, targets):
    return train_network(n, data, targets, perceptron_activation, perceptron_activation, -np.infty)
