import sys
sys.path.append('../../lib/')

import MultiOutLNN
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np

import ReadLensesData
import ConvertData


def to_one_hot(val, m):
    res = np.zeros(m)
    res[val-1] = 1
    return res

data, targets = ReadLensesData.read_data()
data = np.array(data)
targets = np.array([to_one_hot(v, 3) for v in targets])

res = MultiOutLNN.train_lnn(data, targets, 140000, len(data[0]), [30], len(targets[0]), [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation])

rule = MultiOutLNN.ExtractRules(len(data[0]), res, ["OR", "AND"])
print(len(rule))

for i in range(len(rule)):
    print(i)
    print(rule[i])
    print()

print(data)
print(targets)

print(MultiOutLNN.test(rule, data, targets))
