import sys
sys.path.append('../../lib/')

import numpy as np
import ReadLensesData
import ConvertData
from MultiOutLNFN import train_cnf_network, train_dnf_network, run_cnf_network, run_dnf_network
from BooleanFormula import build_cnf, build_dnf

data, raw_targets = ReadLensesData.read_data()

targets = []
for t in raw_targets:
    t_new = [0,0,0]
    t_new[t-1] = 1

    targets.append(t_new)

targets = np.array(targets)
data = np.array(data)

cnf = train_dnf_network(6, data, targets, 700000, 3)

hidden = cnf[0][0]
out = cnf[0][1]
out1 = out[0]
out2 = out[1]
out3 = out[2]

cnf1 = build_dnf(6, (hidden, out1))
cnf2 = build_dnf(6, (hidden, out2))
cnf3 = build_dnf(6, (hidden, out3))

print(cnf1)
print()
print(cnf2)
print()
print(cnf3)
