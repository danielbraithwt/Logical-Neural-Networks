import sys
sys.path.append('../lib/')

from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network, train_perceptron_general_network
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

plt.switch_backend("TkAgg")

n_max = 11
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

def __n_rand_perms(n, size):
    if not n:
        return

    idx = [random.randrange(2**n) for i in range(size)]

    p = []

    for i in idx:
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p

def generateExpressions(n):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), SS)

    return np.array(list(map(lambda x: (inputs, x), outputs)))


def run_experement(func, n, data, targets, q):
    net, loss, time = func(n, data, targets)
    q.put((net, loss, time))

def runExperements(n, data, targets):
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    pcep_g_res = Queue()
    
    cnf_p = Process(target=run_experement, args=(train_cnf_network, n, data, targets, cnf_res))
    dnf_p = Process(target=run_experement, args=(train_dnf_network, n, data, targets, dnf_res))
    pcep_p = Process(target=run_experement, args=(train_perceptron_network, n, data, targets, pcep_res))
    pcep_g_p = Process(target=run_experement, args=(train_perceptron_general_network, n, data, targets, pcep_g_res))

    cnf_p.start()
    dnf_p.start()
    pcep_p.start()
    pcep_g_p.start()

    cnf_p.join()
    dnf_p.join()
    pcep_p.join()
    pcep_g_p.join()

    cnf_net, cnf_loss, cnf_time = cnf_res.get()
    dnf_net, dnf_loss, dnf_time = dnf_res.get()
    pcep_net, pcep_loss, pcep_time = pcep_res.get()
    pcep_g_net, pcep_g_loss, pcep_g_time = pcep_g_res.get()

    print("[T] Training Time: [CNF: " + str(cnf_time) + ", DNF: " + str(dnf_time) + ", PCEP: " + str(pcep_time) + ", PCEP G: " + str(pcep_g_time) + "]")
    print("[T] Training Loss: [CNF: " + str(cnf_loss) + ", DNF: " + str(dnf_loss) + ", PCEP: " + str(pcep_loss) + ", PCEP G: " + str(pcep_g_loss) + "]")
    return cnf_loss, dnf_loss, pcep_loss, pcep_g_loss

if __name__ == "__main__":
    cnf_data = []
    cnf_std = []
    
    dnf_data = []
    dnf_std = []
    
    pcep_data = []
    pcep_std = []

    pcep_g_data = []
    pcep_g_std = []
    
    for n in range(n_start, n_max):
        print("\n\n[!] -- N = " + str(n) + " --")
        print("[*] Generating Expressions")

        cnf_f = open("raw-results/" + str(n) + "-cnf.txt", 'w+')
        dnf_f = open("raw-results/" + str(n) + "-dnf.txt", 'w+')
        pcep_f = open("raw-results/" + str(n) + "-pcep.txt", 'w+')
        pcep_g_f = open("raw-results/" + str(n) + "-pcep-g.txt", 'w+')

        allExpressions = generateExpressions(n)
        
        cnf_means = []
        dnf_means = []
        pcep_means = []
        pcep_g_means = []

        print("[*] Running Experements")
        for e in range(0, len(allExpressions)):
            expression = allExpressions[e]

            data = np.asarray(expression[0].tolist(), dtype=np.float64)
            targets = np.asarray(expression[1].tolist(), dtype=np.float64)

            print("[" + str(e) + "] Experement:")
            print(data)
            print(targets)

            cnf_peformance = []
            dnf_peformance = []
            pcep_peformance = []
            pcep_g_peformance = []

            for r in range(repeat):
                cnf_loss, dnf_loss, pcep_loss, pcep_g_loss = runExperements(n, data, targets)

                cnf_peformance.append(cnf_loss)
                dnf_peformance.append(dnf_loss)
                pcep_peformance.append(pcep_loss)
                pcep_g_peformance.append(pcep_g_loss)


            cnf_m = np.array(cnf_peformance).mean()
            dnf_m = np.array(dnf_peformance).mean()
            pcep_m = np.array(pcep_peformance).mean()
            pcep_g_m = np.array(pcep_g_peformance).mean()

            cnf_means.append(cnf_m)
            dnf_means.append(dnf_m)
            pcep_means.append(pcep_m)
            pcep_g_means.append(pcep_g_m)

            cnf_f.write(str(cnf_peformance) + "\n")
            dnf_f.write(str(dnf_peformance) + "\n")
            pcep_f.write(str(pcep_peformance) + "\n")
            pcep_g_f.write(str(pcep_g_peformance) + "\n")

        cnf_data.append(np.array(cnf_means).mean())
        cnf_std.append(np.array(cnf_means).std())
        
        dnf_data.append(np.array(dnf_means).mean())
        dnf_std.append(np.array(dnf_means).std())
        
        pcep_data.append(np.array(pcep_means).mean())
        pcep_std.append(np.array(pcep_means).std())

        pcep_g_data.append(np.array(pcep_g_means).mean())
        pcep_g_std.append(np.array(pcep_g_means).std())

        cnf_f.close()
        dnf_f.close()
        pcep_f.close()
        pcep_g_f.close()
                    

    # Draw the graph from collected data
    x_axis = np.array(range(n_start, n_max))
    df = np.repeat(SS-1, n_max - n_start)

    plt.plot(x_axis, cnf_data, '-o', color='b', label='CNF')
    plt.errorbar(x_axis, cnf_data, yerr=ss.t.ppf(0.95, df)*cnf_std, color='b')

    plt.plot(x_axis, dnf_data, '-o', color='r', label='DNF')
    plt.errorbar(x_axis, dnf_data, yerr=ss.t.ppf(0.95, df)*dnf_std, color='r')

    plt.plot(x_axis, pcep_data, '-o', color='g', label='Perceptron')
    plt.errorbar(x_axis, pcep_data, yerr=ss.t.ppf(0.95, df)*pcep_std, color='g')
    
    plt.plot(x_axis, pcep_g_data, '-o', color='y', label='Perceptron General')
    plt.errorbar(x_axis, pcep_g_data, yerr=ss.t.ppf(0.95, df)*pcep_std, color='y')

    plt.ylabel("Accuracy Over All Data")
    plt.xlabel("Size Of Expression")
    plt.xlim([n_start-1, n_max + 1])
    plt.legend(loc='best')
    plt.savefig("peformance.png")
