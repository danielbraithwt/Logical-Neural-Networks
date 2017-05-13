import numpy as np
from Neurons import ORNeuron, ANDNeuron

class DNFNetwork():
    def __init__(self, n, c):
        self.n = n
        self.c = c

        d = []
        for i in range(0, c):
            d.append(ANDNeuron(n*2 + 1))

        self.conjunctions = np.array(d)

        if not c == 0:
            self.disjunction = ORNeuron(c + 1)
        else:
            self.disjunction = ORNeuron(n*2 + 1)

    def getOutput(self):
        return self.disjunction

    def getHidden(self):
        return self.conjunctions

    def fowardprop(self, inputs):
        actualIn = self.__convertInputs__(inputs)

        if not self.c == 0:
            dout = [1]
            for d in self.conjunctions:
                dout.append(d.present(actualIn))

            actualIn = dout

        return self.disjunction.present(np.array(actualIn))

    def __convertInputs__(self, inputs):
        actual = [1]

        for i in inputs:
            actual.append(i)
            actual.append(1-i)

        return np.array(actual)

    def __repr__(self):
        s = "Conjunctions -> "
        for d in self.conjunctions:
            s += (str(d) + ", ")

        s += ("\nDisjunction -> " + str(self.disjunction) + "\n")

        return s
