import sys
sys.path.append('../../lib/')

import MultiOutLNN
import numpy as np

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


alarmclock = np.load('alarm-clock.npy')
asparagus = np.load('asparagus.npy')

data = np.concatenate([alarmclock, asparagus])
targets = []
for i in range(len(alarmclock)):
    targets.append([1, 0])
for i in range(len(asparagus)):
    targets.append([0, 1])

targets = np.array(targets)


#print(targets.shape)

X_train, Y_train, X_test, Y_test = split_data(data, targets, 0.7)

X_train = X_train[0:55000] / 255
X_test = X_test[0:55000] / 255

num_inputs = len(X_train[0])
num_outputs = len(Y_train[0])

## CONFIGURATION ##
hidden_layers = []
activations = [MultiOutLNN.noisy_and_activation]
iterations = len(X_train) * 30
## ## ## ## ## ## ##

res = MultiOutLNN.train_lnn(X_train, Y_train[0:55000], iterations, num_inputs, hidden_layers, num_outputs, activations, True)

#rule = MultiOutLNN.ExtractRules(len(X_train[0]), res, ["OR", "AND"])
#print(len(rule))

#for i in range(len(rule)):
#    print(i)
#    print(rule[i])
#    print()

print("Training")
print("Total Number Of Samples: ", len(X_train))
print("Network Wrong: ", MultiOutLNN.run_lnn(X_train, Y_train[0:55000], res, activations, True)/55000)
#print("Rule Set Wrong: ", MultiOutLNN.test(rule, X_train, Y_train))

print()

print("Testing")
print("Total Number Of Samples: ", len(X_test))
print(MultiOutLNN.run_lnn(X_test, Y_test, res, activations, True)/len(X_test))
#print(MultiOutLNN.test(rule, X_test, Y_test))
