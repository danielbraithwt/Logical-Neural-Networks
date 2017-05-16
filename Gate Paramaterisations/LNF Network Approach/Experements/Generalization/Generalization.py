import sys
sys.path.insert(0, '../lib')


from Networks import trainNetwork, __loss
from multiprocessing import Process
import numpy as np

N = 4

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


def runExperement(netType, name, data, targets):
    print("[*] Running Experement: " + name)
    result = open("results/" + name + ".txt", "w+")

    result.write("Expression:\n")
    result.write(str(data) + "\n")
    result.write(str(targets) + "\n")

    # Remove data points starting with 0 and ending with all-1
    for i in range(0, len(data)-1):
        ids = np.random.choice(len(data), 2**N - i, replace=False)
        d = np.take(data, ids)
        t = np.take(targets, ids)

        print("[*] i = " + str(i))
        print("[*] Training With:")
        print("[!]\t" + str(d))
        print("[!]\t" + str(t))

        NET, Loss = trainNetwork(netType, d, t, N, 2**N)

        testData = []
        testTargets = []
        for i in range(0, len(data)):
            if not i in ids:
                testData.append(data[i])
                testTargets.append(targts[i])

        GLoss = __loss(NET, testData, testTargets)
        ALoss = __loss(NET, data, targets)

        result.write("\ni = " + str(i) + "\n")
        result.write("Training Loss: " + str(Loss) + "\n")
        result.write("Generalization Loss: " + str(GLoss) + "\n")
        result.write("Overall Loss: " + str(ALoss) + "\n")

def runExperementThread(netType, name, data, targets):
    return Process(target=runExperement, args=(netType, name, data, targets))

if __name__ == "__main__":
    print("[*] Generating Expressions")
    allExpressions = generateExpressions(N)

    idx = np.random.randint(len(allExpressions), size=1)
    expression = allExpressions[idx,:][0]

    data = expression[0]
    targets = expression[1]

    print("Expression:\n")
    print(str(data) + "\n")
    print(str(targets) + "\n")

    cnfP = runExperementThread("cnf", "CNF", data, targets)
    dnfP = runExperementThread("dnf", "DNF", data, targets)
    PcepP = runExperementThread("perceptron", "Perceptron", data, targets)

    cnfP.start()
    dnfP.start()
    PcepP.start()

    cnfP.join()
    dnfP.join()
    PcepP.join()

    print("[*] Done!")
