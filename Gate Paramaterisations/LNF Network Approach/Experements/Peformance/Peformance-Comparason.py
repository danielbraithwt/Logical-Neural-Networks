import sys
sys.path.insert(0, '../lib')


from Networks import trainNetwork
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue

N_max = 10
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
    DNFResults = Queue()
    CNFResults = Queue()
    PCEPResults = Queue()

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

            print("[" + str(e) + "] Starting Expression")
            for r in range(0, repeat):
                print("[" + str(r) + "] Starting Repition")
                # print("[*] Training CNF")
                cnfP = Process(target=trainNetwork, args=('cnf', data, targets, n, 2**n, CNFResults))
                dnfP = Process(target=trainNetwork, args=('dnf', data, targets, n, 2**n, DNFResults))
                pcepP = Process(target=trainNetwork, args=('perceptron', data, targets, n, 2**n, PCEPResults))

                print("[*] Starting Learning")

                cnfP.start()
                dnfP.start()
                pcepP.start()

                cnfP.join()
                dnfP.join()
                pcepP.join()

                CNFN, CNFLoss = CNFResults.get()
                DNFN, DNFLoss = DNFResults.get()
                PN, PLoss = PCEPResults.get()

                print("[*] Learning Finished")

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
            result.flush()

            print("[" + str(e) + "]: Result - CNF [" + str(CNF_m) + "], DNF [" + str(DNF_m) + "]" + ", P [" + str(P_m) + "]")
        result.close()


    print("[!] DONE!!")
