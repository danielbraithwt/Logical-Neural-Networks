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


targets, examples = read_data()
print(examples[0])
res = LNNWithNot.train_lnn(examples, np.array(targets), 70000, len(examples[0]), [30, 20], 1, [LNNWithNot.noisy_or_activation, LNNWithNot.noisy_and_activation, LNNWithNot.noisy_or_activation])
rule = LNNWithNot.ExtractRules(len(examples[0]), res, ["OR", "AND", "OR"])
print(rule[0])
print(LNNWithNot.test(rule[0], examples, targets))
