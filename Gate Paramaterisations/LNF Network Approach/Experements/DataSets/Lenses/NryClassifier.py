import sys
sys.path.append('../../lib/')

import numpy as np
import ReadLensesData
import ConvertData
import MultiOutLNN
import MultiOutNN

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
cnf_rule_errors = []
dnf_rule_errors = []
# Peform LOE Cross-Validation
for i in range(0, len(data)):
    print(i, " : ", len(data))
    data_p = data[np.arange(len(data)) != i]
    targets_p = targets[np.arange(len(data)) != i]

    print("Training Error")
    cnf = MultiOutLNN.train_lnn(data_p, targets_p, 100000, len(data[0]), [2**len(data[0])], len(targets[0]), [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], True)#train_cnf_network(6, data_p, targets_p, 100000 * 0, 3)
    #print("CNF: ", cnf[2])
    dnf = MultiOutLNN.train_lnn(data_p, targets_p, 100000, len(data[0]), [2**len(data[0])], len(targets[0]), [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], True)#train_dnf_network(6, data_p, targets_p, 100000 * 0, 3)
    #print("DNF: ", dnf[2])
    pcep = MultiOutNN.train_lnn(data_p, targets_p, 100000, len(data[0]), [30], len(targets[0]), True)
    #print("PCEP: ", pcep[2])


    cnf_rule = MultiOutLNN.ExtractRules(len(data[0]), cnf, ["OR", "AND"])
    dnf_rule = MultiOutLNN.ExtractRules(len(data[0]), dnf, ["AND", "OR"])

    cnf_rule_er = MultiOutLNN.test(cnf_rule, data, targets)/len(data)
    cnf_rule_errors.append(cnf_rule_er)
    dnf_rule_er = MultiOutLNN.test(dnf_rule, data, targets)/len(data)
    dnf_rule_errors.append(dnf_rule_er)

    print("Overall Error")
    cnf_er = MultiOutLNN.run_lnn(data, targets, cnf, [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation], True)/len(data)
    print("CNF: ", cnf_er)
    print("CNF Rule: ", cnf_rule_er)
    dnf_er = MultiOutLNN.run_lnn(data, targets, dnf, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], True)/len(data)
    print("DNF: ", dnf_er)
    print("DNF Rule: ", dnf_rule_er)
    pcep_er = MultiOutNN.run_lnn(data_p, targets_p, pcep, len(data[0]), [30], len(targets[0]), True)
    print("PCEP: ", pcep_er)

    cnf_errors.append(cnf_er)
    dnf_errors.append(dnf_er)
    pcep_errors.append(pcep_er)

print("DONE")

cnf_errors = np.array(cnf_errors)
dnf_errors = np.array(dnf_errors)
pcep_errors = np.array(pcep_errors)
cnf_rule_errors = np.array(cnf_rule_errors)
dnf_rule_errors = np.array(dnf_rule_errors)

print()
print("Final Results")
print("CNF: ", cnf_errors.mean(), ", CI: ", conf_interval(cnf_errors))
print("CNF Rule: ", cnf_rule_errors.mean(), ", CI: ", conf_interval(cnf_rule_errors))
print("DNF: ", dnf_errors.mean(), ", CI: ", conf_interval(dnf_errors))
print("DNF Rule: ", dnf_rule_errors.mean(), ", CI: ", conf_interval(dnf_rule_errors))
print("PCEP: ", pcep_errors.mean(), ", CI: ", conf_interval(pcep_errors))

