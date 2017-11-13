import sys
sys.path.append('../../lib/')

import numpy as np
import IrisData
import MultiOutLNN

task_id = None
if len(sys.argv) > 1:
    task_id = sys.argv[1]
    print(task_id)

def conf_interval(data):
    N = len(data)
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)]), np.abs(sorted_estimates[int(0.975 * N)]))
    return conf_interval

def split_data(data, targets, ratio):
    idx = np.random.choice(len(data), int(ratio * len(data)), replace=False)

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for i in range(len(data)):
        if i in idx:
            X_train.append(data[i])
            Y_train.append(targets[i])
        else:
            X_test.append(data[i])
            Y_test.append(targets[i])

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


data, raw_targets = IrisData.read_data()

targets = []
for t in raw_targets:
    t_new = [0,0,0]
    t_new[t-1] = 1

    targets.append(t_new)

targets = np.array(targets)
data = np.array(data)

hidden_layers = [2**4]
activations = [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation]
iterations = len(data) * 2500

data_prime = []
for i in range(len(data)):
    #print()
    #print(data[i])
    ex = np.concatenate([data[i], 1 - data[i]])
    #print(ex)
    data_prime.append(ex)

data_prime = np.array(data_prime)

res = []
for i in range(1):
    print("Experement: ", i, end=' ')
    X_train, Y_train, X_test, Y_test = split_data(data_prime, targets, 0.7)


    net = MultiOutLNN.train_lnn(X_train, Y_train, iterations, 8, np.copy(hidden_layers).tolist(), 3, activations, False)
    wrong = MultiOutLNN.run_lnn(X_test, Y_test, net, activations, False)

    er = wrong/len(X_test)
    print(" -> ", er)
    res.append(er)

    rules = MultiOutLNN.ExtractFuzzyRules(len(X_train[0]), net, ['OR', 'AND'], 0.5, 2, False)

    print()
    print()
    for i in range(len(rules)):
        print(i)
        print(rules[i])
        print()


    print("Training Set")
    wrong = MultiOutLNN.run_lnn(X_train, Y_train, net, activations, False)
    er = wrong/len(X_train)
    train_net_wrong = er
    print("Network Error Rate: ", er)
    wrong = MultiOutLNN.test_fuzzy_rules(rules, X_train, Y_train)
    er = wrong/len(X_train)
    train_rule_wrong = er
    print("Rules Error Rate: ", er)


    print()
    print()

    print("Testing Set")
    wrong = MultiOutLNN.run_lnn(X_test, Y_test, net, activations, False)
    er = wrong/len(X_test)
    test_net_wrong = er
    print("Network Error Rate: ", er)
    wrong = MultiOutLNN.test_fuzzy_rules(rules, X_test, Y_test)
    er = wrong/len(X_test)
    test_rule_wrong = er
    print("Rules Error Rate: ", er)

    if task_id:
        f = open('results-{}.txt'.format(task_id), 'a+')
        f.write(str(test_rule_wrong))
        f.close()

#res = np.array(res)
#print()
#print(res.mean())
#print(conf_interval(res))

#print(train_cnf_network(4, data, targets, 100000, 3))
#print(train_perceptron_network_general(4, data, targets, 700000, 3))
