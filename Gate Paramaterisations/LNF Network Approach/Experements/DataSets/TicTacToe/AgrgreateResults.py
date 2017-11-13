import numpy as np
from functools import reduce

def conf_interval(data):
    N = len(data)
    M = 0#data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval


network_training_error = []
rule_training_error = []
network_testing_error = []
rule_testing_error = []

net_wrong_train = []
rule_wrong_train = []
net_wrong_test = []
rule_wrong_test = []

for i in range(1, 31):
    results = np.load("results-or-{}.npy".format(i))

    network_training_error.append(results[0])
    rule_training_error.append(results[1])
    network_testing_error.append(results[2])
    rule_testing_error.append(results[3])

    net_wrong_train.append(results[4])
    rule_wrong_train.append(results[5])
    net_wrong_test.append(results[6])
    rule_wrong_test.append(results[7])


    
network_training_error = np.array(network_training_error)
rule_training_error = np.array(rule_training_error)
network_testing_error = np.array(network_testing_error)
rule_testing_error = np.array(rule_testing_error)

all_rule_testing = reduce(np.union1d, rule_wrong_test)
overlaps = []
for w in rule_wrong_test:
    overlaps.append(len(w)/len(all_rule_testing))

print(np.array(overlaps).mean())


uniq_network_testing = reduce(np.intersect1d, net_wrong_test)
print(float(len(uniq_network_testing))/float(len(np.union1d(all_rule_testing, uniq_network_testing))))

print("Network Training Error")
print("Mean: ", network_testing_error.mean())
print("CI: ", conf_interval(network_testing_error))
print()
print("Network Testing Error")
print("Mean: ", network_training_error.mean())
print("CI: ", conf_interval(network_training_error))
print()
print()
print("Rule Training Error")
print("Mean: ", rule_training_error.mean())
print("CI: ", conf_interval(rule_training_error))
print()
print("Rule Testing Error")
print("Mean: ", rule_testing_error.mean())
print("CI: ", conf_interval(rule_testing_error))
print()
