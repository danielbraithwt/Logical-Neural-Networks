import sys
sys.path.append('../../lib/')

import numpy as np
import IrisData
import RMEPartitioning
from MultiOutLNFN import train_cnf_network, train_dnf_network, run_cnf_network, run_dnf_network
from MultiOutNN import train_perceptron_network_general, run_perceptron_network_general


def D_equal_width(instances, K):
    #features = features.transpose()
    features = []

    feature_info = []
    for feature in instances.transpose():
        f_min = feature.min()
        f_max = feature.max() + 0.1
        bin_size = (f_max - f_min)/K

        feature_info.append((f_min, f_max, bin_size))

    
    for instance in instances:
        instance_new = np.zeros(len(instance) * K)

        for f_id in range(len(instance)):
            f_val = instance[f_id]
            f_min = feature_info[f_id][0]
            f_max = feature_info[f_id][1]
            f_bin_size = feature_info[f_id][2]
            
            pos = int((f_val - f_min)/f_bin_size)
            instance_new[(K * f_id) + pos] = 1
            
        features.append(instance_new)
    
    return features

def D_equal_frequency(instances, K):
    #features = features.transpose()
    features = []

    feature_info = []
    for feature in instances.transpose():
        limits = []
        sorted_features = np.sort(feature)

        for i in range(1, K-1):
            limits.append(sorted_features[i])
        limits.append(sorted_features[-1])

        feature_info.append(limits)

    
    for instance in instances:
        instance_new = np.zeros(len(instance) * K)

        for f_id in range(len(instance)):
            f_val = instance[f_id]
            f_lim = feature_info[f_id]
            print(f_val)
            print(f_lim)
            pos = 0
            for i in range(1, K):
                if f_val < f_lim[i]:
                    break

                pos += 1
                
            instance_new[(K * f_id) + pos] = 1
            
        features.append(instance_new)
    
    return features


def apply_bin(instance, b, i):
    val = instance[i]
    bin_num = 0
    for idx in range(len(b) - 1):
        if val < b[bin_num]:
            break
        bin_num += 1

    if val > b[-1]:
        bin_num += 1

    return bin_num

def descretize(data, bins, fl):
    d_data = []
    for instance in data:
        new_instance = np.zeros(fl)
        buffer = 0

        for f in range(len(instance)):
            b = bins[f]
            bn = apply_bin(instance, b, f)
            new_instance[buffer + bn] = 1
            buffer += (len(b) + 1)

        d_data.append(new_instance)

    return d_data


raw_data, raw_targets = IrisData.read_data_raw()
p_data = np.concatenate((raw_data, np.expand_dims(raw_targets, 1)), 1)

    
f1_p = RMEPartitioning.partition(p_data, 0, 3)
f2_p = RMEPartitioning.partition(p_data, 1, 3)
f3_p = RMEPartitioning.partition(p_data, 2, 3)
f4_p = RMEPartitioning.partition(p_data, 3, 3)

bins = [f1_p, f2_p, f3_p, f4_p]
data = descretize(raw_data, bins, 12)


targets = []
for t in raw_targets:
    t_new = [0,0,0]
    t_new[t-1] = 1

    targets.append(t_new)

targets = np.array(targets)
data = np.array(data)

cnf = train_cnf_network(len(data[0]), data, targets, 100000, 3)
print(cnf)
