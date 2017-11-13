import sys
sys.path.append('../../lib/')

import LoadFaces
import MultiOutLNN
import tensorflow as tf
import os
import numpy as np


training, test = LoadFaces.read_data()

X_train = np.array(training[0])
Y_train = np.array(training[1])

X_test = np.array(test[0])
Y_test = np.array(test[1])

num_inputs = len(X_train[0])
num_outputs = len(Y_train[0])

## CONFIGURATION ##
hidden_layers = [4]
activations = [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation]
iterations = len(X_train) * 30
## ## ## ## ## ## ##

res = MultiOutLNN.train_lnn(X_train, Y_train, iterations, num_inputs, hidden_layers, num_outputs, ["OR", "AND"], activations, True)

#rule = MultiOutLNN.ExtractRules(len(X_train[0]), res, ["OR", "AND"])
#print(len(rule))

#for i in range(len(rule)):
#    print(i)
#    print(rule[i])
#    print()

print("Training")
print("Total Number Of Samples: ", len(X_train))
print("Network Wrong: ", MultiOutLNN.run_lnn(X_train, Y_train, res, num_inputs, [8], num_outputs, activations, True))
#print("Rule Set Wrong: ", MultiOutLNN.test(rule, X_train, Y_train))

print()

print("Testing")
print("Total Number Of Samples: ", len(X_test))
print(MultiOutLNN.run_lnn(X_test, Y_test, res, num_inputs, [8], num_outputs, activations, True))
#print(MultiOutLNN.test(rule, X_test, Y_test))
