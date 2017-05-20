from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import matplotlib.pyplot as plt
import scipy.stats as ss

n_max = 4
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
    cnf_data = []
    cnf_std = []
    
    dnf_data = []
    dnf_std = []
    
    pcep_data = []
    pcep_std = []
    
    for n in range(n_start, n_max):
        print("\n\n[!] -- N = " + str(n) + " --")
        print("[*] Generating Expressions")

        cnf_f = open("raw-results/" + str(n) + "-cnf.txt", 'w+')
        dnf_f = open("raw-results/" + str(n) + "-dnf.txt", 'w+')
        pcep_f = open("raw-results/" + str(n) + "-pcep.txt", 'w+')

        allExpressions = generateExpressions(n)
        idx = np.random.randint(len(allExpressions), size=SS)
        expressions = allExpressions[idx,:]

        cnf_means = []
        dnf_means = []
        pcep_means = []

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


            cnf_m = np.array(cnf_peformance).mean()
            dnf_m = np.array(dnf_peformance).mean()
            pcep_m = np.array(pcep_peformance).mean()

            cnf_means.append(cnf_m)
            dnf_means.append(dnf_m)
            pcep_means.append(pcep_m)

            cnf_f.writeline(str(cnf_means))
            dnf_f.writeline(str(dnf_means))
            pcep_f.writeline(str(pcep_means))

        cnf_data.append(np.array(cnf_means).mean())
        cnf_std.append(np.array(cnf_means).std())
        
        dnf_data.append(np.array(dnf_means).mean())
        dnf_std.append(np.array(dnf_means).std())
        
        pcep_data.append(np.array(pcep_means).mean())
        pecp_std.append(np.array(pcep_means).std())
                    

    # Draw the graph from collected data
    x_axis = np.array(range(n_start, n_max))
    df = np.repeat(SS-1, n_max - n_start)
    
    plt.errorbar(x_axis, cnf_data, yerr=ss.t.ppf(0.95, df)*cnf_std)
    plt.errorbar(x_axis, dnf_data, yerr=ss.t.ppf(0.95, df)*dnf_std)
    plt.errorbar(x_axis, pecp_data, yerr=ss.t.ppf(0.95, df)*pcep_std)
