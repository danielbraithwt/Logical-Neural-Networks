import SPECTData
import sys
sys.path.append('../../lib/')

import numpy as np
import MultiOutLNN

train, test = SPECTData.read_data()

activations  =[MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation]

net = MultiOutLNN.train_lnn(train[0], train[1], 1000 * len(train[0]), len(train[0][0]), [30], 1, [MultiOutLNN.noisy_and_activation, MultiOutLNN.noisy_or_activation], True)
wrong = MultiOutLNN.run_lnn(test[0], test[1], net, activations, True)

er = wrong/len(test[0])
print(er)
