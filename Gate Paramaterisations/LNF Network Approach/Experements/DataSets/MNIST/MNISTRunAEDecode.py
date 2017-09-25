import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def transform_weights(weights):
    return np.log(1 + np.exp(-weights))

def noisy_or_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)

    print(t_w.shape)
    z = np.add(np.matmul(inputs, np.transpose(t_w)), t_b)
    print(z.shape)
    return 1 - np.exp(-z)

def noisy_and_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)

    z = np.add(np.matmul(1 - inputs, np.transpose(t_w)), t_b)
    return np.exp(-z)

def apply_network(network, activations, iput, addNot):
    prev_out = iput
    for idx in range(len(network)):
        #print()
        #print(prev_out.shape)
        #print(network[idx].shape)
        if addNot:
            #print(prev_out)
            prev_out = np.concatenate([prev_out, 1 - prev_out], axis=1)
        
        
        layer = network[idx]
        act = activations[idx]
        
        w = layer
        #b = layer[1]
        #print(b.shape)

        #print(prev_out.shape)
        out = act(prev_out, w, 0)
        #print(out.shape)
        prev_out = out

    return prev_out

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + "/")
print(tf.gfile.Exists(dir_path + "/"))
mnist = input_data.read_data_sets(dir_path + "/", one_hot=False)

X_train = mnist.train.images
#Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels


network = np.load('mnist-autoencoder.npy')
activations = [noisy_or_activation, noisy_and_activation, noisy_and_activation]

vec = np.array([[1,1,1,1,0,0,0,0,0,0]])
decoder = np.array([network[1]])[0]
encoder = np.array([network[0]])[0]

print(len(decoder[0]))

numbers = []
for i in range(len(Y_test)):
    if Y_test[i] in numbers:
        continue

    numbers.append(Y_test[i])
    enc = apply_network(encoder, activations, np.array([X_test[i]]), True)
    print(enc)
    decoded = apply_network(decoder, np.flip(activations, 0), enc, True)
    print(Y_test[i], " : ", enc)

    img = np.reshape(decoded, (28, 28))
    hm = plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.savefig("Image{}.png".format(Y_test[i]))
    plt.clf()


print(decoder.shape)
decoded = apply_network(decoder, np.flip(activations, 0), vec, True)
#print(decoded)
img = np.reshape(decoded, (28, 28))

hm = plt.imshow(img, cmap='gray', interpolation='nearest')
plt.savefig("Image.png")
plt.clf()
