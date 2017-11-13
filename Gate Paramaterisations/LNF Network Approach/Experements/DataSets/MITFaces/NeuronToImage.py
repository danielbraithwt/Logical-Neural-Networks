import tensorflow as tf
import random
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from PIL import Image

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


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
    data = np.zeros((weights.shape[0]))

    print(hidden_weights.shape)
    for j in range(int(hidden_weights.shape[1]/2)):
        #print(j, " : ", hidden_weights.shape[1]/2)
        j_not = int(hidden_weights.shape[1]/2) + j
        #print(j, " : ", j_not)

        res = (hidden_weights[i][j_not])# - hidden_weights[i][j_not])

        data[j] = res #hidden_weights[i][j_not]#((1 - hidden_weights[i][j]) * 255, 0, (1 - hidden_weights[i][j_not]) * 255)

    image = np.zeros((19, 19))

    for a in range(19):
        for b in range(19):
            #print(a * 19 + b)
            image[a,b] = data[a * 19 + b]
            
    #print(image)

    imgs.append(np.copy(image))
    hm = plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.colorbar(hm, ticks=[-1,0,1])
    plt.savefig("Neuron-{}.png".format(i))
    plt.clf()
    #img = Image.fromarray(image, 'Grey')
    #img.save("Neuron-{}.png".format(i))
    #img.show()


imgs = np.array(imgs)

digit_imgs = []
output_weights = network[1][0]
for i in range(len(output_weights)):
    wts = sigmoid(output_weights[i])
    wts_pos = wts[0:4]

    print(wts_pos)

    count = 0
    img = np.zeros((19, 19))
    for c in range(len(wts_pos)):
        #if wts_pos[c] == 0:
        img = img + wts_pos[c] * imgs[c]
        count += 1

    #print(img)
    img = img/count
    digit_imgs.append(img)

    hm = plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.colorbar(hm)
    plt.savefig("Digit-{}.png".format(i))
    plt.clf()
    
#print(np.round(sigmoid(network[1][0]), 3))
