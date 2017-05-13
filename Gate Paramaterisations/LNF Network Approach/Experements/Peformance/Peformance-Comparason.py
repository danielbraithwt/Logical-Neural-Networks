import sys
sys.path.insert(0, '../lib')


from Networks import trainNetwork
import numpy as np
import threading

N_max = 4
SS = 5
repeat = 1

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

for n in range(2, N_max):
    print("\n\n[!] -- N = " + str(n) + " --")
    result = open("results/" + str(n) + "-num-vars.txt", "w+")
    print("[*] Generating Expressions")
    allExpressions = generateExpressions(n)
    # print(allExpressions)
    idx = np.random.randint(len(allExpressions), size=SS)
    expressions = allExpressions[idx,:]

    print("[*] Running Experements")
    for e in range(0, len(expressions)):
        expression = expressions[e]

        data = expression[0]
        targets = expression[1]

        CNFPeformance = np.array([])
        DNFPeformance = np.array([])
        PecpPeformance = np.array([])

        print("[" + str(e) + "]: Starting Expression")
        for r in range(0, repeat):
            print("[" + str(r) + "]: Starting Repition")
            print("[*] Training CNF")
            CNFN, CNFLoss = trainNetwork('cnf', data, targets, n, 2**n)
            print("[*] Training DNF")
            DNFN, DNFLoss = trainNetwork('dnf', data, targets, n, 2**n)
            print("[*] Training Perceptron")
            PN, PLoss = trainNetwork('perceptron', data, targets, n, 2**n)

            CNFPeformance = np.append(CNFPeformance, CNFLoss)
            DNFPeformance = np.append(DNFPeformance, DNFLoss)
            PecpPeformance = np.append(PecpPeformance, PLoss)

        CNF_m = CNFPeformance.mean()
        DNF_m = DNFPeformance.mean()
        P_m = PecpPeformance.mean()

        result.writelines(str(n))
        result.writelines(str(expression))
        result.writelines(str(CNFPeformance) + " : " + str(CNF_m) + " : " + str(np.var(CNFPeformance)))
        result.writelines(str(DNFPeformance) + " : " + str(DNF_m) + " : " + str(np.var(DNFPeformance)))
        result.writelines(str(PecpPeformance) + " : " + str(P_m) + " : " + str(np.var(PecpPeformance)))
        result.writelines("\n")

        print("[" + str(e) + "]: Result - CNF [" + str(CNF_m) + "], DNF [" + str(DNF_m) + "]" + ", P [" + str(P_m) + "]")
    result.close()


print("[!] DONE!!")
