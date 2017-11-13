import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../lib/')
import MultiOutLNN


network = np.load('network.npy')

rules = MultiOutLNN.ExtractFuzzyRules(8, network, ['AND', 'OR'], 0.1, 2, False)

print()
print()
for i in range(len(rules)):
    print(i)
    print(rules[i])
    print()


