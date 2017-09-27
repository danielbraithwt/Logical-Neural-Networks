import os
import numpy as np

def conf_interval(data):
    N = len(data)
    M = 0#data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval

d = '.'
potential_results = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]

results_dirs = [res for res in potential_results if 'Model' in res]

for result_dir in results_dirs:
    training_data = []
    testing_data = []
    
    for filename in os.listdir(result_dir):
        content = open(os.path.join('{}/'.format(result_dir), filename), 'r')
        for line in content:
            #print(line)
            results = line.split(':')
            training_data.append(float(results[0].strip()))
            testing_data.append(float(results[1].strip()))


    training_data = np.array(training_data)
    testing_data = np.array(testing_data)

    print()
    print(result_dir)
    #print("Training: ")
    #print("\tMean: ", training_data.mean())
    #print("\tCI: ", conf_interval(training_data))
    print("Testing: ")
    print("\tMean: ", testing_data.mean())
    print("\tCI: ", conf_interval(testing_data))
    print(testing_data)
