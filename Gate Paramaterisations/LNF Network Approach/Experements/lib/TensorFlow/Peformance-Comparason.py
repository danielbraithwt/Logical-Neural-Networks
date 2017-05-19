from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import matplotlib.pyplot as plt

n_max = 10
n_start = 2
SS = 5
repeat = 5

def __perms(n):
    if not n:
        return

    p = []

    for i in range(0, 2**n):
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = list(map(lambda x: int(x), list(s)))
        p.append(s_prime)

    return p

def generateExpressions(n):
    inputs = __perms(n)
    outputs = __perms(len(inputs))

    return np.array(list(map(lambda x: (inputs, x), outputs)))


if __name__ == "__main__":
    for n in range(n_start, n_max):
        print("\n\n[!] -- N = " + str(n) + " --")
        print("[*] Generating Expressions")

        allExpressions = generateExpressions(n)
        idx = np.random.randint(len(allExpressions), size=SS)
        expressions = allExpressions[idx,:]

        print("[*] Running Experements")
        for e in range(0, len(expressions)):
            expression = expressions[e]

            data = expression[0]
            targets = expression[1]

            cnf_peformance = []
            dnf_peformance = []
            pcep_peformance = []

            for r in range(repeat):
                cnf_loss, dnf_loss, pcep_loss = runExperements(n, data, targets)

                cnf_peformance.append(cnf_loss)
                dnf_peformance.append(dnf_loss)
                pecp_peformance.append(pcep_loss)

            
            
                    
