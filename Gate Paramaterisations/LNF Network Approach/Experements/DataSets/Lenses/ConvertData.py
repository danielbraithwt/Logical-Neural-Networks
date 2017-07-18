import numpy as np

def convert_to_binary(targets, t):
    new_targets = []
    for tar in targets:
        if t == tar:
            new_targets.append(1)
        else:
            new_targets.append(0)

    return np.array(new_targets)
