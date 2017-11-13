import sys
sys.path.append('../../lib/')

import numpy as np
import MultiOutLNN

def read_data():
    unparsed = []
    f = open('balance-scale.data', 'r+')
    for line in f:
        unparsed.append(line.strip().split(','))
    f.close()

    
    targets = []
    examples = []

    for e in unparsed:
        if e[0] == 'L':
            targets.append(to_one_hot(0, 3))
        elif e[0] == 'B':
            targets.append(to_one_hot(1, 3))
        else:
            targets.append(to_one_hot(2, 3))

        example = np.concatenate([to_one_hot(int(e[i])-1, 5) for i in range(1, len(e))])
        examples.append(example)

    return targets, examples

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


acts = [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation]
acts_name = ["AND", "OR"]

targets, examples = read_data()

X_train, Y_train, X_test, Y_test = split_data(examples, targets, 0.7)

res = MultiOutLNN.train_lnn(np.array(X_train), np.array(Y_train), 600000, len(examples[0]), [60], 3, acts_name, acts, False)

##rule = LNNWithNot.ExtractRules(len(examples[0]), res, acts_name)
##print(rule)


##print("-- Evaluating --")
##print("Training Set, Size = ", len(X_train))
print("Network Wrong: ", MultiOutLNN.run_lnn(X_train, Y_train, res,  len(examples[0]), [30], 1, acts, False))
##print("Rule Wrong: ", LNNWithNot.test(rule[0], X_train, Y_train))
##print()
##print("Testing Set, Size = ", len(X_test))
##print("Network Wrong: ", LNNWithNot.run_lnn(X_test, Y_test, res,  len(examples[0]), [30], 1, acts))
##print("Rule Wrong: ", LNNWithNot.test(rule[0], X_test, Y_test))


