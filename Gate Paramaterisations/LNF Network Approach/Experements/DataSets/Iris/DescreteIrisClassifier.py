import sys
sys.path.append('../../lib/')

import numpy as np
import IrisData
from MultiOutLNFN import train_cnf_network, train_dnf_network, run_cnf_network, run_dnf_network
from MultiOutNN import train_perceptron_network_general, run_perceptron_network_general


def D_equal_width(features, K):
    features = features.transpose()
    for f in features:
        f_min = f.min()
        f_max = f.max()

        step = (f_max - f_min)/K

    
    return features

raw_data, raw_targets = IrisData.read_data_raw()
print(D_equal_width(raw_data, 4))

##targets = []
##for t in raw_targets:
##    t_new = [0,0,0]
##    t_new[t-1] = 1
##
##    targets.append(t_new)
##
##targets = np.array(targets)
##data = np.array(data)
