import numpy as np
from Neurons import Perceptron

class PerceptronNetwork():
    def __init__(self, n, c):
        self.n = n
        self.c = c

        d = []
        for i in range(0, c):
            d.append(Perceptron(n*2 + 1))

        self.disjunctions = np.array(d)

        if not c == 0:
            self.conjunction = Perceptron(c + 1)
        else:
            self.conjunction = Perceptron(n*2 + 1)

    def getHidden(self):
        return self.disjunctions

    def getOutput(self):
        return self.conjunction

    def fowardprop(self, inputs):
        actualIn = self.__convertInputs__(inputs)

        if not self.c == 0:
            dout = [1]
            for d in self.disjunctions:
                dout.append(d.present(actualIn))

            actualIn = dout

        return self.conjunction.present(np.array(actualIn))

    def __convertInputs__(self, inputs):
        actual = [1]

        for i in inputs:
            actual.append(i)
            actual.append(1-i)

        return np.array(actual)

    def __repr__(self):
        s = "Hidden -> "
        for d in self.disjunctions:
            s += (str(d) + ", ")

        s += ("\Output -> " + str(self.conjunction) + "\n")

        return s
