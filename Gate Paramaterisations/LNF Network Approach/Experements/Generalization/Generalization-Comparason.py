import sys
sys.path.append('../lib/')

import MultiOutLNN
import MultiOutNN
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

n_start = 6
n_max = 7
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


def train_cnf_network(data, targets):
    res = MultiOutLNN.train_lnn(data, targets, 3000 * len(data), len(data[0]), [2**n], 1, [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], False)
    wrong = MultiOutLNN.run_lnn(data, targets, res, [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], False)
    er = float(wrong)/float(len(data))
    return res, er

def train_dnf_network(data, targets):
    res = MultiOutLNN.train_lnn(data, targets, 3000 * len(data), len(data[0]), [2**n], 1, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], False)
    wrong = MultiOutLNN.run_lnn(data, targets, res, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], False)
    er = float(wrong)/float(len(data))
    return res, er


def train_perceptron_network_general(data, targets):
    res = MultiOutNN.train_lnn(data, targets, 3000 * len(data), len(data[0]), [n, n], 1, False)
    wrong = MultiOutNN.run_lnn(data, targets, res, False)
    er = float(wrong)/float(len(data))
    return res, er


def run_cnf_network(data, targets, res):
    wrong = MultiOutLNN.run_lnn(data, targets, res, [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], False)
    er = float(wrong)/float(len(data))
    return er

def run_dnf_network(data, targets, res):
    wrong = MultiOutLNN.run_lnn(data, targets, res, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], False)
    er = float(wrong)/float(len(data))
    return er


def run_perceptron_network_general(data, targets, res):
    wrong = MultiOutNN.run_lnn(data, targets, res, False)
    er = float(wrong)/float(len(data))
    return er

def train(func, n, data, targets, q):
    net, er = func(data, targets)
    q.put((net, er))

def run(func, net, n, data, targets, q):
    loss = func(data, targets, net)
    q.put(loss)

def run_training(n, data, targets):
    data, targets = transform(data, targets)
    print(data)
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    pcep_g_res = Queue()
    
    cnf_p = Process(target=train, args=(train_cnf_network, n, data, targets, cnf_res))
    dnf_p = Process(target=train, args=(train_dnf_network, n, data, targets, dnf_res))
    #pcep_p = Process(target=train, args=(train_perceptron_network, n, data, targets, pcep_res))
    pcep_g_p = Process(target=train, args=(train_perceptron_network_general, n, data, targets, pcep_g_res))

    cnf_p.start()
    dnf_p.start()
    #pcep_p.start()
    pcep_g_p.start()

    cnf_p.join()
    dnf_p.join()
    #pcep_p.join()
    pcep_g_p.join()

    cnf_net, cnf_loss = cnf_res.get()
    dnf_net, dnf_loss = dnf_res.get()
   # pcep_net, pcep_loss, pcep_time = pcep_res.get()
    pcep_g_net, pcep_g_loss = pcep_g_res.get()

    #print("[T] Training Time: [CNF: " + str(cnf_time) + ", DNF: " + str(dnf_time) + ", PCEP: " + str(pcep_time) + ", PCEP G: " + str(pcep_g_time) + "]")
    #print("[T] Training Loss: [CNF: " + str(cnf_loss) + ", DNF: " + str(dnf_loss) + ", PCEP: " + str(pcep_loss) + ", PCEP G: " + str(pcep_g_loss) + "]")
    #print("[T] Training Time: [CNF: " + str(cnf_time) + ", DNF: " + str(dnf_time) + ", PCEP G: " + str(pcep_g_time) + "]")
    print("[T] Training Loss: [CNF: " + str(cnf_loss) + ", DNF: " + str(dnf_loss) + ", PCEP G: " + str(pcep_g_loss) + "]")
    return cnf_net, dnf_net, pcep_g_net

def run_test(n, cnf, dnf, pcep_g, data, targets):
    data, targets = transform(data, targets)
    cnf_res = Queue()
    dnf_res = Queue()
    #pcep_res = Queue()
    pcep_g_res = Queue()
    
    cnf_p = Process(target=run, args=(run_cnf_network, cnf, n, data, targets, cnf_res))
    dnf_p = Process(target=run, args=(run_dnf_network, dnf, n, data, targets, dnf_res))
    #pcep_p = Process(target=run, args=(run_perceptron_network, pcep, n, data, targets, pcep_res))
    pcep_g_p = Process(target=run, args=(run_perceptron_network_general, pcep_g, n, data, targets, pcep_g_res))

    cnf_p.start()
    dnf_p.start()
    #pcep_p.start()
    pcep_g_p.start()

    cnf_p.join()
    dnf_p.join()
    #pcep_p.join()
    pcep_g_p.join()

    cnf_loss= cnf_res.get()
    dnf_loss = dnf_res.get()
    #pcep_loss = pcep_res.get()
    pcep_g_loss = pcep_g_res.get()

    print("[T] Training Loss: [CNF: " + str(cnf_loss) + ", DNF: " + str(dnf_loss) + ", PCEP G: " + str(pcep_g_loss) + "]")
    return cnf_loss, dnf_loss, pcep_g_loss

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

def conf_interval(data):
    N = len(data)
    M = data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval


def transform(data, targets):
    data = np.array([np.concatenate((x, 1-x), 0) for x in data])
    targets = np.array([[x] for x in targets])

    return data, targets

def plot(x, y, ci, c, name):
    ci=np.transpose(np.array(ci))
    plt.errorbar(x, y, yerr=ci, marker='o', color=c, label=name)
    
def run_experement(n, data, targets):

    
    # Remove data points starting with 0 and ending with all-1
    cnf_means = []
    cnf_ci = []
    dnf_means = []
    dnf_ci = []
    #pcep_means = []
    #pcep_std = []
    pcep_g_means = []
    pcep_g_ci = []
    for i in range(1, len(data)):
        print("[*] i = " + str(i))

        cnf_peformance = []
        dnf_peformance = []
        #pcep_peformance = []
        pcep_g_peformance = []


        for r in range(repeat):
            training_data, training_targets, test_data, test_targets = compute_data(data, targets, i)
            print(len(training_data))
            print(training_data, training_targets)
            cnf, dnf, pcep_g = run_training(n, training_data.tolist(), training_targets)

            cnf_loss, dnf_loss, pcep_g_loss = run_test(n, cnf, dnf, pcep_g, data.tolist(), targets)

            cnf_peformance.append(cnf_loss)
            dnf_peformance.append(dnf_loss)
            #pcep_peformance.append(pcep_loss)
            pcep_g_peformance.append(pcep_g_loss)

        cnf_means.append(np.array(cnf_peformance).mean())
        cnf_ci.append(conf_interval(np.array(cnf_peformance)))
        
        dnf_means.append(np.array(dnf_peformance).mean())
        dnf_ci.append(conf_interval(np.array(dnf_peformance)))
        
        #pcep_means.append(np.array(pcep_peformance).mean())
        #pcep_std.append(np.array(pcep_peformance).std())
        
        pcep_g_means.append(np.array(pcep_g_peformance).mean())
        pcep_g_ci.append(conf_interval(np.array(pcep_g_peformance)))


    x_axis = np.array(range(1, len(data)))
    df = np.repeat(repeat-1, len(x_axis))

    plot(x_axis, cnf_means, cnf_ci, 'b', 'CNF')
    plot(x_axis, dnf_means, dnf_ci, 'r', 'DNF')
    plot(x_axis, pcep_g_means, pcep_g_ci, 'y', 'Perceptron')


    plt.ylabel("Error (Over All Data)")
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
