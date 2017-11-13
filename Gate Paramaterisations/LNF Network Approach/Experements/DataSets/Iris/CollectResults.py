import os
import numpy as np

def conf_interval(data):
    N = len(data)
    M = 0#data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval

testing_data_rule = []

for filename in os.listdir('./results/'):
    content = open(os.path.join('./results/', filename), 'r+')
    training_results = content.readline().strip()

    testing_data_rule.append(float(training_results))
    
    content.close()

testing_data_rule = np.array(testing_data_rule)

print(testing_data_rule.mean())
print(conf_interval(testing_data_rule))
