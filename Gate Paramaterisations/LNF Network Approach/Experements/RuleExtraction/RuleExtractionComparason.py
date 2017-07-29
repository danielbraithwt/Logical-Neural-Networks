import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
sys.path.append('../lib/')

from RealSpaceLNFNetwork import train_cnf_network, train_dnf_network
from NeuralNetwork import train_perceptron_network, train_perceptron_network_general
import BooleanFormula
import MLPNRuleExtraction

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

def generateExpressions(n, num):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), num)

    return np.array(list(map(lambda x: (inputs, x), outputs)))

def extractMLPNRules(N, res):
    hidden_weights = res[1][0][0]
    hidden_bias = res[1][0][1]
    output_weights = res[1][1][0]
    output_bias = res[1][1][1]

    rule = MLPNRuleExtraction.ExtractRules(N, hidden_weights, hidden_bias, output_weights, output_bias)
    return rule

def conf_interval(data):
    N = len(data)
    M = data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval


def plot(x, y, ci, c, name):
    ci=np.transpose(np.array(ci))
    plt.errorbar(x, y, yerr=ci, marker='o', color=c, label=name)

N_min = 2
N_max = 9
repeat = 5

mlpn_data = []
mlpn_ci = []

cnf_data = []
cnf_ci = []

dnf_data = []
dnf_ci = []

for N in range(N_min, N_max+1):
    print("N: ", N)
    mlpn_wrong = []
    cnf_wrong = []
    dnf_wrong = []
    
    for i in range(repeat):
        print("\t", i, " -> ")
        expression = generateExpressions(N, 1)[0]
        data = expression[0]
        targets = expression[1]

        cnf_net = train_cnf_network(N, data, targets, 0)#100000)
        print("\tCNF Acuracy: ", cnf_net[2])
        dnf_net = train_dnf_network(N, data, targets, 0)#100000)
        print("\tDNF Acuracy: ", dnf_net[2])
        pcep_net = train_perceptron_network_general(N, data, targets, 0)#100000)
        print("\tMLPN Acuracy: ", pcep_net[2])

        mlpn_rule = extractMLPNRules(N, pcep_net)
        cnf_rule = BooleanFormula.build_cnf(N, cnf_net[0])
        dnf_rule = BooleanFormula.build_dnf(N, dnf_net[0])

        print()

        mlpn_rule_wrong = BooleanFormula.test_cnf(mlpn_rule, data, targets)
        print("\tMLPN Wrong: ", mlpn_rule_wrong)
        cnf_rule_wrong = BooleanFormula.test_cnf(cnf_rule, data, targets)
        print("\tCNF Wrong: ", cnf_rule_wrong)
        dnf_rule_wrong = BooleanFormula.test_cnf(dnf_rule, data, targets)
        print("\tDNF Wrong: ", dnf_rule_wrong)

        mlpn_wrong.append(mlpn_rule_wrong)
        cnf_wrong.append(cnf_rule_wrong)
        dnf_wrong.append(dnf_rule_wrong)

    mlpn_wrong = np.array(mlpn_wrong)
    cnf_wrong = np.array(cnf_wrong)
    dnf_wrong = np.array(dnf_wrong)
    
    mlpn_data.append(mlpn_wrong.mean())
    mlpn_ci.append(conf_interval(mlpn_wrong))
    cnf_data.append(cnf_wrong.mean())
    cnf_ci.append(conf_interval(cnf_wrong))
    dnf_data.append(dnf_wrong.mean())
    dnf_ci.append(conf_interval(dnf_wrong))

x_axis = np.array(range(N_min, N_max+1))
plot(x_axis, cnf_data, cnf_ci, 'b', 'CNF')
plot(x_axis, dnf_data, dnf_ci, 'r', 'DNF')
plot(x_axis, mlpn_data, mlpn_ci, 'y', 'Perceptron')

plt.ylabel("Rule Accuracy")
plt.xlabel("Size of Expression")
#plt.xlim([0, len(data)])
plt.legend(loc='best')
plt.savefig("rule-extract-comparason.png")
plt.clf()
