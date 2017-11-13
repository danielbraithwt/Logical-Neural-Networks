import os
import numpy as np

def conf_interval(data):
    N = len(data)
    M = 0#data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval

training_data_net = []
training_data_rule = []
testing_data_net = []
testing_data_rule = []

for filename in os.listdir('./results/'):
    content = open(os.path.join('./results/', filename), 'r+')
    training_results = content.readline().split(':')
    testing_results = content.readline().split(':')

    training_data_net.append(float(training_results[0]))
    training_data_rule.append(float(training_results[1]))
    testing_data_net.append(float(testing_results[0]))
    testing_data_rule.append(float(testing_results[1]))
    
    content.close()

training_data_net = np.array(training_data_net)
training_data_rule = np.array(training_data_rule)
testing_data_net = np.array(testing_data_net)
testing_data_rule = np.array(testing_data_rule)

print(training_data_net.mean())
print(conf_interval(training_data_net))
print()
print(training_data_rule.mean())
print(conf_interval(training_data_rule))

print()
print("--")
print()

print(testing_data_net.mean())
print(conf_interval(testing_data_net))
print()
print(testing_data_rule.mean())
print(conf_interval(testing_data_rule))
