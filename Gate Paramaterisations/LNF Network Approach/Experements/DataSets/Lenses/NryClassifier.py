import sys
sys.path.append('../../lib/')

import numpy as np
import ReadLensesData
import ConvertData
from MultiOutLNFN import train_cnf_network, train_dnf_network, run_cnf_network, run_dnf_network
from NeuralNetwork import train_perceptron_network_general, run_perceptron_network_general

def conf_interval(data):
    N = len(data)
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)]), np.abs(sorted_estimates[int(0.975 * N)]))
    return conf_interval


data, raw_targets = ReadLensesData.read_data()

targets = []
for t in raw_targets:
    t_new = [0,0,0]
    t_new[t-1] = 1

    targets.append(t_new)

targets = np.array(targets)
data = np.array(data)

cnf_errors = []
dnf_errors = []
pcep_errors = []
# Peform LOE Cross-Validation
for i in range(0, len(data)):
    print(i, " : ", len(data))
    data_p = data[np.arange(len(data)) != i]
    targets_p = targets[np.arange(len(data)) != i]

    print("Training Error")
    cnf = train_cnf_network(6, data_p, targets_p, 100000, 3)
    print("CNF: ", cnf[2])
    dnf = train_dnf_network(6, data_p, targets_p, 100000, 3)
    print("DNF: ", dnf[2])
    pcep = train_perceptron_network_general(6, data_p, targets_p, 100000, 3)
    print("PCEP: ", pcep[2])


    print("Overall Error")
    cnf_er = run_cnf_network(6, data, targets, cnf[1])
    print("CNF: ", cnf_er)
    dnf_er = run_dnf_network(6, data, targets, dnf[1])
    print("DNF: ", dnf_er)
    pcep_er = run_perceptron_network_general(6, data, targets, pcep[1])
    print("PCEP: ", pcep_er)

    cnf_errors.append(cnf_er)
    dnf_errors.append(dnf_er)
    pcep_errors.append(pcep_er)


cnf_errors = np.array(cnf_errors)
dnf_errors = np.array(dnf_errors)
pcep_errors = np.array(pcep_errors)

print()
print("Final Results")
print("CNF: ", cnf_errors.mean(), ", CI: ", conf_interval(cnf_errors))
print("DNF: ", dnf_errors.mean(), ", CI: ", conf_interval(dnf_errors))
print("PCEP: ", pcep_errors.mean(), ", CI: ", conf_interval(pcep_errors))
