import sys
sys.path.append('../../lib/')

import numpy as np
import IrisData
from MultiOutLNFN import train_cnf_network, train_dnf_network, run_cnf_network, run_dnf_network
from MultiOutNN import train_perceptron_network_general, run_perceptron_network_general


data, raw_targets = IrisData.read_data()

targets = []
for t in raw_targets:
    t_new = [0,0,0]
    t_new[t-1] = 1

    targets.append(t_new)

targets = np.array(targets)
data = np.array(data)

print(train_cnf_network(4, data, targets, 700000, 3))
