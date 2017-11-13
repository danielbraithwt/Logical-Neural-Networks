import sys
sys.path.append('../../lib/')

import MultiOutLNN
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + "/")
print(tf.gfile.Exists(dir_path + "/"))

mnist = input_data.read_data_sets(dir_path + "/", one_hot=True)

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

network = np.load('network.npy')

rules = MultiOutLNN.ExtractFuzzyRules(len(X_train[0]), network, ["OR", "AND"], 0.5, 2)


print()
print()
for i in range(len(rules)):
    print(i)
    print(rules[i])
    print()

##wrong = MultiOutLNN.test_fuzzy_rules(rules, X_train, Y_train)
##er = wrong/len(X_train)
##train_rule_wrong = er
##print("Rules Error Rate: ", er)

wrong = MultiOutLNN.test_fuzzy_rules(rules, X_test, Y_test)
er = wrong/len(X_test)
train_rule_wrong = er
print("Rules Error Rate: ", er)
