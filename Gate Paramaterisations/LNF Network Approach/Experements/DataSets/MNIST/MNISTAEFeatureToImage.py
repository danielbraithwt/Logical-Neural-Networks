import tensorflow as tf
import random
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from PIL import Image

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

ae = np.load('mnist-autoencoder.npy')
enc = ae[0][0]
dec = ae[1][0]

print(enc[0])

imgs = []
for i in range(len(enc)-1):
    weights = enc[i]#sigmoid(enc[i])
    print(weights.shape)
    for l in range(len(weights)):
        w = weights[l]
        min_val = w.min()
        max_val = np.abs(w.max())
        for i in range(len(w)):
            if w[i] < 0:
                w[i] = w[i]/min_val
            else:
                w[i] = w[i]/max_val
        print(w.shape)
        #data = np.zeros(int(len(w)/2))
        #for j in range(int(len(w)/2)):
        data = np.zeros(int(len(w)))
        for j in range(int(len(w))):
            j_not = int(len(w)/2) + j

            res = (w[j])# - hidden_weights[i][j_not])
            #print(res)

            data[j] = res #hidden_weights[i][j_not]#((1 - hidden_weights[i][j]) * 255, 0, (1 - hidden_weights[i][j_not]) * 255)

        image = np.zeros((28, 28))

        for a in range(28):
            for b in range(28):
                image[a,b] = data[a * 28 + b]
            
    #print(image)

        imgs.append(np.copy(image))
        hm = plt.imshow(image, cmap='gray', interpolation='nearest',
            vmax=abs(image).max(), vmin=-abs(image).max())
        plt.colorbar(hm, ticks=[-1,0,1])
        plt.savefig("Feature-{}.png".format(l))
        plt.clf()
