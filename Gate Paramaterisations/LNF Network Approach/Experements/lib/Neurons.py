import numpy as np

class Perceptron():
    def __init__(self, n):
        self.n = n
        self.weights = np.random.rand(n)

    def getWeights(self):
        return self.weights

    def present(self, inputs):
        z = np.sum(np.multiply(inputs, self.weights))
        return 1 /(1 + np.power(np.e, -z))

    def updateWeights(self, g):
        self.weights -= g

    def setGrad(self, g):
        self.grad = g

    def applyGrad(self):
        self.updateWeights(self.grad)
        self.grad = None

    def __repr__(self):
        return str(self.weights[0:len(self.weights)])

class ORNeuron():
    def __init__(self, n):
        self.n = n
        self.weights = np.random.rand(n)

    def getWeights(self):
        return self.weights

    def present(self, inputs):
        z = np.sum(np.multiply(inputs, self.weights))
        return 1 - np.power(np.e, -z)

    def updateWeights(self, g):
        self.weights -= g

        for i in range(0, len(self.weights)):
            if self.weights[i] < 0:
                self.weights[i] = 0

    def setGrad(self, g):
        self.grad = g

    def applyGrad(self):
        self.updateWeights(self.grad)
        self.grad = None

    def __repr__(self):
        return str(self.weights[1:len(self.weights)])


class ANDNeuron():
    def __init__(self, n):
        self.weights = np.random.rand(n)

    def getWeights(self):
        return self.weights

    def present(self, inputs):
        i = 1.0 - inputs
        i[0] = 1.0

        z = np.sum(np.multiply(i, self.weights))
        return np.power(np.e, -z)

    def updateWeights(self, g):
        self.weights -= g

        for i in range(0, len(self.weights)):
            if self.weights[i] < 0:
                self.weights[i] = 0

    def setGrad(self, g):
        self.grad = g

    def applyGrad(self):
        self.updateWeights(self.grad)
        self.grad = None

    def __repr__(self):
        return str(self.weights[1:len(self.weights)])
