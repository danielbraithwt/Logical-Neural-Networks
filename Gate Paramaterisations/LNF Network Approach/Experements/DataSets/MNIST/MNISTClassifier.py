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

print(len(X_train[0]))

num_inputs = len(X_train[0])
num_outputs = len(Y_train[0])

## CONFIGURATION ##
hidden_layers = []
activations = [MultiOutLNN.noisy_and_activation]
iterations = len(X_train) * 30
## ## ## ## ## ## ##

res = MultiOutLNN.train_lnn(X_train, Y_train, iterations, num_inputs, hidden_layers, num_outputs, ["AND", "OR"], activations, True)

#rule = MultiOutLNN.ExtractRules(len(X_train[0]), res, ["OR", "AND"])
#print(len(rule))

#for i in range(len(rule)):
#    print(i)
#    print(rule[i])
#    print()

print("Training")
print("Total Number Of Samples: ", len(X_train))
print("Network Wrong: ", MultiOutLNN.run_lnn(X_train, Y_train, res, num_inputs, [30], num_outputs, activations, True))
#print("Rule Set Wrong: ", MultiOutLNN.test(rule, X_train, Y_train))

print()

print("Testing")
print("Total Number Of Samples: ", len(X_test))
print(MultiOutLNN.run_lnn(X_test, Y_test, res, num_inputs, [30], num_outputs, activations, True))
#print(MultiOutLNN.test(rule, X_test, Y_test))
