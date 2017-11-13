import VehicleData
import sys
sys.path.append('../../lib/')

import numpy as np
import MultiOutLNN

#task_id = sys.argv[1]

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

    return [np.array(X_train), np.array(Y_train)], [np.array(X_test), np.array(Y_test)]

examples, targets = VehicleData.read_data()

train, test = split_data(examples, targets, 0.7)

activations = [MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation,  MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation,  MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_or_activation, MultiOutLNN.noisy_and_activation]

net = MultiOutLNN.train_lnn(train[0], train[1], 500 * len(train[0]), len(train[0][0]), [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 30, 30, 30, 30, 30, 30, 30, 30], 4, activations, True)


print(len(train[0][0]))
print(train[0][0])
rules = MultiOutLNN.ExtractFuzzyRules(len(train[0][0]), net, ['OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'OR', 'AND'], 0.1, 1)

print()
print()
for i in range(len(rules)):
    print(i)
    #print(len(rules[i]))
    print()

print("Training Set")
wrong = MultiOutLNN.run_lnn(train[0], train[1], net, activations, True)
er = wrong/len(train[0])
train_net_wrong = er
print("Network Error Rate: ", er)
wrong = MultiOutLNN.test_fuzzy_rules(rules, train[0], train[1])
er = wrong/len(train[0])
train_rule_wrong = er
print("Rules Error Rate: ", er)


print()
print()

print("Testing Set")
wrong = MultiOutLNN.run_lnn(test[0], test[1], net, activations, True)
er = wrong/len(test[0])
test_net_wrong = er
print("Network Error Rate: ", er)
wrong = MultiOutLNN.test_fuzzy_rules(rules, test[0], test[1])
er = wrong/len(test[0])
test_rule_wrong = er
print("Rules Error Rate: ", er)

#f = open('result-{}.txt'.format(task_id), 'a+')
#f.write("{}:{}\n".format(train_net_wrong, train_rule_wrong))
#f.write("{}:{}\n".format(test_net_wrong, test_rule_wrong))
#f.close()
