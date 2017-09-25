import tensorflow as tf
import random
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from PIL import Image

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


activations = ["OR", "AND", "AND"]

network = np.load('network.npy')

hidden_layer = network[0]
print(hidden_layer)
hidden_weights = hidden_layer[0]
hidden_weights = sigmoid(hidden_weights)
print(hidden_weights)

#
#hidden_weights = special.expit(hidden_weights)

imgs = []

# For each hidden weight
for i in range(len(hidden_weights)):
    weights = hidden_weights[i]
    print(weights)
    data = np.zeros(int(hidden_weights.shape[1]/2))

    for j in range(int(hidden_weights.shape[1]/2)):
        j_not = int(hidden_weights.shape[1]/2) + j

        res = (hidden_weights[i][j])# - hidden_weights[i][j_not])

        data[j] = res #hidden_weights[i][j_not]#((1 - hidden_weights[i][j]) * 255, 0, (1 - hidden_weights[i][j_not]) * 255)

    image = np.zeros((28, 28))

    for a in range(28):
        for b in range(28):
            image[a,b] = data[a * 28 + b]
            
    #print(image)

    imgs.append(np.copy(image))
    hm = plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.colorbar(hm, ticks=[0,1])
    plt.savefig("Layer0-Neuron-{}.png".format(i))
    plt.clf()
    #img = Image.fromarray(image, 'Grey')
    #img.save("Neuron-{}.png".format(i))
    #img.show()


imgs = np.array(imgs)
for l in range(1,len(network)):
    digit_imgs = []
    output_weights = network[l][0]
    for i in range(len(output_weights)):
        wts = sigmoid(output_weights[i])
        wts_pos = wts[0:int(len(wts)/2)]

        #print(wts_pos)
        print("Layer {}, Neuron {}".format(l, i))
        #print(len(wts_pos))
        #print(len(imgs))
        

        count = 0
        img = np.zeros((28, 28))
        for c in range(len(wts_pos)):
            if np.round(wts_pos[c], 1) < 1.1:
                img = img + wts_pos[c] * imgs[c]
                count += 1

        #print(img)
        if not count == 0:
            img = img/count
        digit_imgs.append(img)

        hm = plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.colorbar(hm, ticks=[0,1])
        plt.savefig("Layer{}-Neuron-{}.png".format(l, i))
        plt.clf()

    imgs = np.copy(digit_imgs)
    
#print(np.round(sigmoid(network[1][0]), 3))
