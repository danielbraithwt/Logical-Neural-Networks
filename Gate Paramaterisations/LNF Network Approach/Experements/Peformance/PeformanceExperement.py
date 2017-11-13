import sys
sys.path.append('../lib/')

import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import random

import MultiOutLNN
import MultiOutNN

SS = 1

n = int(sys.argv[1])
task_id = sys.argv[2]

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


def train_cnf_network(data, targets, q):
    res = MultiOutLNN.train_lnn(data, targets, 5000 * len(data), len(data[0]), [2**n], 1, [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], False)
    wrong = MultiOutLNN.run_lnn(data, targets, res, [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], False)
    er = float(wrong)/float(len(data))
    q.put(er)

def train_dnf_network(data, targets, q):
    res = MultiOutLNN.train_lnn(data, targets, 5000 * len(data), len(data[0]), [2**n], 1, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], False)
    wrong = MultiOutLNN.run_lnn(data, targets, res, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], False)
    er = float(wrong)/float(len(data))
    q.put(er)

def train_pcep_network(data, targets, q):
    res = MultiOutNN.train_lnn(data, targets, 5000 * len(data), len(data[0]), [2**n], 1, False)
    wrong = MultiOutNN.run_lnn(data, targets, res, False)
    er = float(wrong)/float(len(data))
    q.put(er)

def train_pcep_g_network(data, targets, q):
    res = MultiOutNN.train_lnn(data, targets, 5000 * len(data), len(data[0]), [n, n], 1, False)
    wrong = MultiOutNN.run_lnn(data, targets, res, False)
    er = float(wrong)/float(len(data))
    q.put(er)
    

def run_experement(data, targets):
    cnf_res = Queue()
    dnf_res = Queue()
    pcep_res = Queue()
    pcep_g_res = Queue()

    cnf_p = Process(target=train_cnf_network, args=(data, targets, cnf_res))
    dnf_p = Process(target=train_dnf_network, args=(data, targets, dnf_res))
    pcep_p = Process(target=train_pcep_network, args=(data, targets, pcep_res))
    pcep_g_p = Process(target=train_pcep_g_network, args=(data, targets, pcep_g_res))

    cnf_p.start()
    dnf_p.start()
    pcep_p.start()
    pcep_g_p.start()

    cnf_p.join()
    dnf_p.join()
    pcep_p.join()
    pcep_g_p.join()    

    cnf_er = cnf_res.get()
    dnf_er = dnf_res.get()
    pcep_er = pcep_res.get()
    pcep_g_er = pcep_g_res.get()

    return cnf_er, dnf_er, pcep_er, pcep_g_er


allExpressions = generateExpressions(n)
expression = allExpressions[0]

data = np.asarray(expression[0].tolist(), dtype=np.float64)
targets = np.asarray(expression[1].tolist(), dtype=np.float64)

data = np.array([np.concatenate((x, 1-x), 0) for x in data])
targets = np.array([[x] for x in targets])
#print(targets)
    
cnf_er, dnf_er, pcep_er, pcep_g_er = run_experement(data, targets)
print(cnf_er)
print(dnf_er)
print(pcep_er)
print(pcep_g_er)

f= open('./results/result-{}-{}.txt'.format(n, task_id), 'a+')
f.write('{}:{}:{}:{}'.format(cnf_er, dnf_er, pcep_er, pcep_g_er))
f.close()
