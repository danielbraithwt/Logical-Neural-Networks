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
for i in range(len(enc)):
    weights = sigmoid(enc[i])
    data = np.zeros(len(weights))
    for j in range(int(len(weights)/2)):
        j_not = int(len(weights)/2) + j

        res = (weights[j])# - hidden_weights[i][j_not])
        print(res)

        data[j] = res #hidden_weights[i][j_not]#((1 - hidden_weights[i][j]) * 255, 0, (1 - hidden_weights[i][j_not]) * 255)

    image = np.zeros((28, 28))

    for a in range(28):
        for b in range(28):
            image[a,b] = data[a * 28 + b]
            
    #print(image)

    imgs.append(np.copy(image))
    hm = plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.colorbar(hm, ticks=[0,1])
    plt.savefig("Feature-{}.png".format(i))
    plt.clf()
