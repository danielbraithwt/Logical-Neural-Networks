import sys
sys.path.append('../../lib/')

import numpy as np
import ReadLensesData
import ConvertData
from RealSpaceLNFNetwork import train_cnf_network, train_dnf_network, run_cnf_network, run_dnf_network

def conf_interval(data):
    N = len(data)
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)]), np.abs(sorted_estimates[int(0.975 * N)]))
    return conf_interval

data, targets = ReadLensesData.read_data()
data = np.array(data)
targets = np.array(targets)

errors = []
# Peform LOE Cross-Validation
for i in range(0, len(data)):
    print(i, " : ", len(data))
    data_p = data[np.arange(len(data)) != i]
    targets1 = ConvertData.convert_to_binary(targets, 1)
    targets2 = ConvertData.convert_to_binary(targets, 2)
    targets3 = ConvertData.convert_to_binary(targets, 3)

    targets1_p = targets1[np.arange(len(data)) != i]
    targets2_p = targets2[np.arange(len(data)) != i]
    targets3_p = targets3[np.arange(len(data)) != i]

    hard = train_cnf_network(6, data_p, np.array(targets1_p), 90000)
    print("Hard: ", hard[2])
    soft = train_cnf_network(6, data_p, np.array(targets2_p), 90000)
    print("Soft: ", soft[2])
    no = train_cnf_network(6, data_p, np.array(targets3_p), 90000)
    print("NO: ", no[2])

    hard_er = run_cnf_network(6, data.tolist(), targets1, hard[1])
    soft_er = run_cnf_network(6, data.tolist(), targets2, soft[1])
    no_er = run_cnf_network(6, data.tolist(), targets3, no[1])

    avg = (hard_er + soft_er + no_er)/3.0
    print("Average Error: ", avg)
    errors.append(avg)


errors = np.array(errors)
print("Mean Error: ", errors.mean())
print("Error CI: ", conf_interval(errors))
