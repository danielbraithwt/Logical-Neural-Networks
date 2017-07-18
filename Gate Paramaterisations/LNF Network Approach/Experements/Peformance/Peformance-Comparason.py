import sys
sys.path.append('../lib/')

from RealSpaceLNFNetwork import train_cnf_network, train_dnf_network
from NeuralNetwork import train_perceptron_network, train_perceptron_network_general
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

n_max = 10
n_start = 2
SS = 5
repeat = 1

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
    _, net, loss, time = func(n, data, targets, 100000)
    q.put((net, loss, time))

def conf_interval(data):
    N = len(data)
    M = data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval

def runExperements(n, data, targets):
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    pcep_g_res = Queue()
    
    cnf_p = Process(target=run_experement, args=(train_cnf_network, n, data, targets, cnf_res))
    dnf_p = Process(target=run_experement, args=(train_dnf_network, n, data, targets, dnf_res))
    pcep_p = Process(target=run_experement, args=(train_perceptron_network, n, data, targets, pcep_res))
    pcep_g_p = Process(target=run_experement, args=(train_perceptron_network_general, n, data, targets, pcep_g_res))

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

def plot(x, y, ci, c, name):
    ci=np.transpose(np.array(ci))
    plt.errorbar(x, y, yerr=ci, marker='o', color=c, label=name)
    

if __name__ == "__main__":
    cnf_data = []
    cnf_ci = []
    
    dnf_data = []
    dnf_ci = []
    
    pcep_data = []
    pcep_ci = []

    pcep_g_data = []
    pcep_g_ci = []
    
    for n in range(n_start, n_max):
        print("\n\n[!] -- N = " + str(n) + " --")
        print("[*] Generating Expressions")

        allExpressions = generateExpressions(n)
        

        cnf_peformance = []
        dnf_peformance = []
        pcep_peformance = []
        pcep_g_peformance = []

        print("[*] Running Experements")
        for e in range(0, len(allExpressions)):
            expression = allExpressions[e]

            data = np.asarray(expression[0].tolist(), dtype=np.float64)
            targets = np.asarray(expression[1].tolist(), dtype=np.float64)

            print("[" + str(e) + "] Experement:")
            print(data)
            print(targets)

            for r in range(repeat):
                cnf_loss, dnf_loss, pcep_loss, pcep_g_loss = runExperements(n, data, targets)
                #cnf_loss, dnf_loss = runExperements(n, data, targets)

                cnf_peformance.append(cnf_loss)
                dnf_peformance.append(dnf_loss)
                pcep_peformance.append(pcep_loss)
                pcep_g_peformance.append(pcep_g_loss)

        
        cnf_peformance = np.array(cnf_peformance)
        dnf_peformance = np.array(dnf_peformance)
        pcep_peformance = np.array(pcep_peformance)
        pcep_g_peformance = np.array(pcep_g_peformance)

        cnf_data.append(cnf_peformance.mean())
        cnf_ci.append(conf_interval(cnf_peformance))
        
        dnf_data.append(dnf_peformance.mean())
        dnf_ci.append(conf_interval(dnf_peformance))

        pcep_data.append(pcep_peformance.mean())
        pcep_ci.append(conf_interval(pcep_peformance))
        
        pcep_g_data.append(pcep_g_peformance.mean())
        pcep_g_ci.append(conf_interval(pcep_g_peformance))
                    

    # Draw the graph from collected data
    x_axis = np.array(range(n_start, n_max))
    #df = np.repeat(SS-1, n_max - n_start)

    print(cnf_ci)

    plot(x_axis, cnf_data, cnf_ci, 'b', 'CNF')
    plot(x_axis, dnf_data, dnf_ci, 'r', 'DNF')
    plot(x_axis, pcep_data, pcep_ci, 'g', 'Perceptron (Same Config)')
    plot(x_axis, pcep_g_data, pcep_g_ci, 'y', 'Perceptron')

    plt.ylabel("Error")
    plt.xlabel("Size Of Expression")
    plt.xlim([n_start-1, n_max + 1])
    plt.legend(loc='best')
    plt.savefig("all-peformance.png")

    plt.clf()

    plot(x_axis, cnf_data, cnf_ci, 'b', 'CNF')
    plot(x_axis, dnf_data, dnf_ci, 'r', 'DNF')

    plt.ylabel("Error")
    plt.xlabel("Size Of Expression")
    plt.xlim([n_start-1, n_max + 1])
    plt.legend(loc='best')
    plt.savefig("lnfn-peformance.png")
