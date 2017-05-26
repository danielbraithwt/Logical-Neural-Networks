import sys
sys.path.append('../lib/')

from NetworkHelpers import train_cnf_network, train_dnf_network, train_perceptron_network, train_perceptron_network_general, run_cnf_network, run_dnf_network, run_perceptron_network, train_perceptron_general_network, run_perceptron_general_network
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

n_start = 2
n_max = 11
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

def generateExpressions(n, size):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), size)

    return np.array(list(map(lambda x: (inputs, x), outputs)))


def train(func, n, data, targets, q):
    net, loss, time = func(n, data, targets)
    q.put((net, loss, time))

def run(func, net, n, data, targets, q):
    loss = func(n, net, data, targets)
    q.put(loss)

def run_training(n, data, targets):
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    pcep_g_res = Queue()
    
    cnf_p = Process(target=train, args=(train_cnf_network, n, data, targets, cnf_res))
    dnf_p = Process(target=train, args=(train_dnf_network, n, data, targets, dnf_res))
    pcep_p = Process(target=train, args=(train_perceptron_network, n, data, targets, pcep_res))
    pcep_g_p = Process(target=train, args=(train_perceptron_general_network, n, data, targets, pcep_g_res))

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
    return cnf_net, dnf_net, pcep_net, pcep_g_net

def run_test(n, cnf, dnf, pcep, pcep_g, data, targets):
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    pcep_g_res = Queue()
    
    cnf_p = Process(target=run, args=(run_cnf_network, cnf, n, data, targets, cnf_res))
    dnf_p = Process(target=run, args=(run_dnf_network, dnf, n, data, targets, dnf_res))
    pcep_p = Process(target=run, args=(run_perceptron_network, pcep, n, data, targets, pcep_res))
    pcep_g_p = Process(target=run, args=(run_perceptron_general_network, pcep_g, n, data, targets, pcep_g_res))

    cnf_p.start()
    dnf_p.start()
    pcep_p.start()
    pcep_g_p.start()

    cnf_p.join()
    dnf_p.join()
    pcep_p.join()
    pcep_g_p.join()

    cnf_loss= cnf_res.get()
    dnf_loss = dnf_res.get()
    pcep_loss = pcep_res.get()
    pcep_g_loss = pcep_g_res.get()

    print("[T] Training Loss: [CNF: " + str(cnf_loss) + ", DNF: " + str(dnf_loss) + ", PCEP: " + str(pcep_loss) + ", PCEP G: " + str(pcep_g_loss) + "]")
    return cnf_loss, dnf_loss, pcep_loss, pcep_g_loss

def compute_data(data, targets, i):
    ids = np.random.choice(len(data), 2**n - i, replace=False)
    d = np.take(data, ids)
    t = np.take(targets, ids)

    testData = []
    testTargets = []
    for i in range(0, len(data)):
        if not i in ids:
            testData.append(data[i])
            testTargets.append(targets[i])

    return d, t, testData, testTargets

def run_experement(n, data, targets):
    cnf_f = open("raw-results/" + str(n) + "-cnf.txt", 'w+')
    dnf_f = open("raw-results/" + str(n) + "-dnf.txt", 'w+')
    pcep_f = open("raw-results/" + str(n) + "-pcep.txt", 'w+')
    pcep_g_f = open("raw-results/" + str(n) + "-pcep-g.txt", 'w+')
    
    # Remove data points starting with 0 and ending with all-1
    cnf_means = []
    cnf_std = []
    dnf_means = []
    dnf_std = []
    pcep_means = []
    pcep_std = []
    pcep_g_means = []
    pcep_g_std = []
    for i in range(1, len(data)):
        print("[*] i = " + str(i))

        cnf_peformance = []
        dnf_peformance = []
        pcep_peformance = []
        pcep_g_peformance = []


        for r in range(repeat):
            training_data, training_targets, test_data, test_targets = compute_data(data, targets, i)
            print(len(training_data))
            print(training_data, training_targets)
            cnf, dnf, pcep, pcep_g = run_training(n, training_data.tolist(), training_targets.tolist())

            cnf_loss, dnf_loss, pcep_loss, pcep_g_loss = run_test(n, cnf, dnf, pcep, pcep_g, data.tolist(), targets.tolist())

            cnf_peformance.append(cnf_loss)
            dnf_peformance.append(dnf_loss)
            pcep_peformance.append(pcep_loss)
            pcep_g_peformance.append(pcep_g_loss)

        cnf_means.append(np.array(cnf_peformance).mean())
        cnf_std.append(np.array(cnf_peformance).std())
        
        dnf_means.append(np.array(dnf_peformance).mean())
        dnf_std.append(np.array(dnf_peformance).std())
        
        pcep_means.append(np.array(pcep_peformance).mean())
        pcep_std.append(np.array(pcep_peformance).std())
        
        pcep_g_means.append(np.array(pcep_g_peformance).mean())
        pcep_g_std.append(np.array(pcep_g_peformance).std())


    cnf_f.write(str(cnf_means) + "\n")
    dnf_f.write(str(dnf_means) + "\n")
    pcep_f.write(str(pcep_means) + "\n")
    pcep_g_f.write(str(pcep_g_means) + "\n")

    x_axis = np.array(range(1, len(data)))
    df = np.repeat(repeat-1, len(x_axis))

    plt.plot(x_axis, cnf_means, '-o', color='b', label='CNF')
    plt.errorbar(x_axis, cnf_means, yerr=ss.t.ppf(0.95, df)*cnf_std, color='b')

    plt.plot(x_axis, dnf_means, '-o', color='r', label='DNF')
    plt.errorbar(x_axis, dnf_means, yerr=ss.t.ppf(0.95, df)*dnf_std, color='r')

    plt.plot(x_axis, pcep_means, '-o', color='g', label='Perceptron')
    plt.errorbar(x_axis, pcep_means, yerr=ss.t.ppf(0.95, df)*pcep_std, color='g')
    
    plt.plot(x_axis, pcep_g_means, '-o', color='y', label='Perceptron General')
    plt.errorbar(x_axis, pcep_g_means, yerr=ss.t.ppf(0.95, df)*pcep_g_std, color='y')

    plt.ylabel("Accuracy Over All Data")
    plt.xlabel("Number Of Training Examples Removed")
    plt.xlim([0, len(data)])
    plt.legend(loc='best')
    plt.savefig(str(n) + "-generalization.png")
    plt.clf()
                                            

        
if __name__ == "__main__":
    print("[*] Generating Expressions")

    for n in range(n_start, n_max):
        expression = generateExpressions(n, 1)[0]

        data = expression[0]
        targets = expression[1]

        print("Expression:\n")
        print(str(data) + "\n")
        print(str(targets) + "\n")


        run_experement(n, data, targets)

        print("[*] Done!")
