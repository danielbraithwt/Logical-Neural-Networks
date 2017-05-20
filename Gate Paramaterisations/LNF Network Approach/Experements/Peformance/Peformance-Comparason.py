import sys
sys.path.append('../lib/')

from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

plt.switch_backend("TkAgg")

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

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p

def generateExpressions(n):
    inputs = __perms(n)
    outputs = __perms(len(inputs))

    return np.array(list(map(lambda x: (inputs, x), outputs)))


def run_experement(func, n, data, targets, q):
    net, loss, time = func(n, data, targets)
    q.put((net, loss, time))

def runExperements(n, data, targets):
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    
    cnf_p = Process(target=run_experement, args=(train_cnf_network, n, data, targets, cnf_res))
    dnf_p = Process(target=run_experement, args=(train_dnf_network, n, data, targets, dnf_res))
    pcep_p = Process(target=run_experement, args=(train_perceptron_network, n, data, targets, pcep_res))

    cnf_p.start()
    dnf_p.start()
    pcep_p.start()

    cnf_p.join()
    dnf_p.join()
    pcep_p.join()

    cnf_net, cnf_loss, cnf_time = cnf_res.get()
    dnf_net, dnf_loss, dnf_time = dnf_res.get()
    pcep_net, pcep_loss, pcep_time = pcep_res.get()

    print("[T] Training Time: [CNF: " + str(cnf_time) + ", DNF: " + str(dnf_time) + ", PCEP: " + str(pcep_time))
    return cnf_loss, dnf_loss, pcep_loss

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

            data = np.asarray(expression[0].tolist(), dtype=np.float64)
            targets = np.asarray(expression[1].tolist(), dtype=np.float64)

            print("[" + str(e) + "] Experement:")
            print(data)
            print(targets)

            cnf_peformance = []
            dnf_peformance = []
            pcep_peformance = []

            for r in range(repeat):
                cnf_loss, dnf_loss, pcep_loss = runExperements(n, data, targets)

                cnf_peformance.append(cnf_loss)
                dnf_peformance.append(dnf_loss)
                pcep_peformance.append(pcep_loss)


            cnf_m = np.array(cnf_peformance).mean()
            dnf_m = np.array(dnf_peformance).mean()
            pcep_m = np.array(pcep_peformance).mean()

            cnf_means.append(cnf_m)
            dnf_means.append(dnf_m)
            pcep_means.append(pcep_m)

            cnf_f.write(str(cnf_means) + "\n")
            dnf_f.write(str(dnf_means) + "\n")
            pcep_f.write(str(pcep_means) + "\n")

        cnf_data.append(np.array(cnf_means).mean())
        cnf_std.append(np.array(cnf_means).std())
        
        dnf_data.append(np.array(dnf_means).mean())
        dnf_std.append(np.array(dnf_means).std())
        
        pcep_data.append(np.array(pcep_means).mean())
        pcep_std.append(np.array(pcep_means).std())

        cnf_f.close()
        dnf_f.close()
        pcep_f.close()
                    

    # Draw the graph from collected data
    x_axis = np.array(range(n_start, n_max))
    df = np.repeat(SS-1, n_max - n_start)

    plt.plot(x_axis, cnf_data, '-o', color='b', label='CNF')
    plt.errorbar(x_axis, cnf_data, yerr=ss.t.ppf(0.95, df)*cnf_std, color='b')

    plt.plot(x_axis, dnf_data, '-o', color='r', label='DNF')
    plt.errorbar(x_axis, dnf_data, yerr=ss.t.ppf(0.95, df)*dnf_std, color='r')

    plt.plot(x_axis, pcep_data, '-o', color='g', label='Perceptron')
    plt.errorbar(x_axis, pcep_data, yerr=ss.t.ppf(0.95, df)*pcep_std, color='g')
    
                 
    plt.savefig("peformance.png")
