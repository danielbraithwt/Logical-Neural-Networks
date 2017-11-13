import sys
sys.path.append('../../lib/')

import numpy as np
import LNNWithNot

def read_data():
    unparsed = []
    f = open('tic-tac-toe.data', 'r+')
    for line in f:
        unparsed.append(line.strip().split(','))
    f.close()

    
    targets = []
    examples = []

    for e in unparsed:
        if e[-1] == 'positive':
            targets.append(1.0)
        else:
            targets.append(0.0)

        example = np.concatenate([to_one_hot(to_numerical(e[i]), 3) for i in range(0, len(e)-1)])
        examples.append(example)

    return targets, examples

def to_numerical(val):
    if val == 'x':
        return 0
    elif val == 'o':
        return 1
    else:
        return 2

def to_one_hot(val, m):
    vec = np.zeros(m)
    vec[val] = 1
    return vec

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

    return X_train, Y_train, X_test, Y_test


task_id = sys.argv[1]

acts = [LNNWithNot.noisy_or_activation, LNNWithNot.noisy_and_activation]
acts_name = ["OR", "AND"]

targets, examples = read_data()

X_train, Y_train, X_test, Y_test = split_data(examples, targets, 0.7)

res = LNNWithNot.train_lnn(X_train, np.array(Y_train), 600000, len(examples[0]), [100], 1, acts)

rule = LNNWithNot.ExtractRules(len(examples[0]), res, acts_name)
print(rule[0])
print(rule[0].get_literals())
print(len(rule[0].get_literals()))

wrong_net_train, id_wrong_net_train = LNNWithNot.run_lnn(X_train, Y_train, res,  len(examples[0]), [30], 1, acts)
wrong_rule_train, id_wrong_rule_train = LNNWithNot.test(rule[0], X_train, Y_train)

wrong_net_test, id_wrong_net_test = LNNWithNot.run_lnn(X_test, Y_test, res,  len(examples[0]), [30], 1, acts)
wrong_rule_test, id_wrong_rule_test = LNNWithNot.test(rule[0], X_test, Y_test)

net_train_err = wrong_net_train/len(X_train)
rule_train_err = wrong_rule_train/len(X_train)

net_test_err = wrong_net_test/len(X_test)
rule_test_err = wrong_rule_test/len(X_test)

res = [net_train_err, rule_train_err, net_test_err, rule_test_err, np.array(id_wrong_net_train), np.array(id_wrong_rule_train), np.array(id_wrong_net_test), np.array(id_wrong_rule_test)]
res = np.array(res, dtype=object)

#np.save('results-or-{}'.format(task_id), res)

#print("-- Evaluating --")
#print("Training Set, Size = ", len(X_train))
#print("Network Wrong: ", LNNWithNot.run_lnn(X_train, Y_train, res,  len(examples[0]), [30], 1, acts))
#print("Rule Wrong: ", LNNWithNot.test(rule[0], X_train, Y_train))
#print()
#print("Testing Set, Size = ", len(X_test))
#print("Network Wrong: ", LNNWithNot.run_lnn(X_test, Y_test, res,  len(examples[0]), [30], 1, acts))
#print("Rule Wrong: ", LNNWithNot.test(rule[0], X_test, Y_test))
