import sys
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


N_min = 2
N_max = 9
repeat = 5

pcep_data = []
cnf_data = []
dnf_data = []

for N in range(N_min, N_max+1):
    mlpn_wrong = []
    cnf_wrong = []
    dnf_wrong = []
    
    for i in range(repeat):
        expression = generateExpressions(N, 1)[0]
        data = expression[0]
        targets = expression[1]

        cnf_net = train_cnf_network(N, data, targets, 100000)
        dnf_net = train_dnf_network(N, data, targets, 100000)
        pcep_net = train_perceptron_network_general(N, data, targets, 100000)

        mlpn_rule = extractMLPNRules(N, pcep_net)
        cnf_rule = BooleanFormula.build_cnf(N, cnf_net[0])
        dnf_rule = BooleanFormula.build_dnf(N, dnf_net[0])

        mlpn_rule_wrong = BooleanFormula.test_cnf(mlpn_rule, data, targets)
        cnf_rule_wrong = BooleanFormula.test_cnf(cnf_rule, data, targets)
        dnf_rule_wrong = BooleanFormula.test_cnf(dnf_rule, data, targets)

        mlpn_wrong.append(mlpn_rule_wrong)
        cnf_wrong.append(cnf_rule_wrong)
        dnf_wrong.append(dnf_rule_wrong)
