import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, rgb2gray, label2rgb

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
mnist = input_data.read_data_sets(dir_path)

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.images

model = np.load('lnn-network.npy')

def softmax(vals):
    ex = np.exp(vals)
    return ex/np.sum(ex)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def noisy_or_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)

    z = np.add(np.dot(inputs, np.transpose(t_w)), t_b)
    return 1 - np.exp(-z)

def noisy_and_activation(inputs, weights, bias):
    t_w = transform_weights(weights)
    t_b = transform_weights(bias)

    z = np.add(np.dot(1 - inputs, np.transpose(t_w)), t_b)
    return np.exp(-z)

def transform_weights(weights):
    return np.log((1 + np.exp(-weights)))

def apply_model(m, img):

    prev_out = img
    prev_out = np.transpose(np.expand_dims(prev_out, 1))
    for l in range(len(m)):
        #print(prev_out.shape)
        prev_out = np.concatenate([prev_out, 1-prev_out], axis=1)
        #print(prev_out.shape)
        layer = m[l]
        #print(layer)
        weights = layer[0]
        biases = layer[1]

        #out = #np.add(np.dot(prev_out, np.transpose(weights)), biases)
        out = activations[l](prev_out, weights, biases)
        #out = np.transpose(out)
        prev_out = out

    #print(prev_out)
    return prev_out[0]/np.sum(prev_out)
        

        

def predict_fn(imgs):
    predictions = []
    for img in imgs:
        #print(img.shape)
        
        img = rgb2gray(img)
        #print(img.shape)
        img = np.reshape(img, (-1))
        #print(img.shape)
        predictions.append(apply_model(model, img))

    #print(predictions)
    return predictions
    


def convert_vec(img):
    img = np.reshape(img, (28, 28))
    img = gray2rgb(img)
    #print(img)
    return img


activations = [noisy_and_activation]#, noisy_and_activation, noisy_and_activation]

print(predict_fn([convert_vec(X_train[0])]))


explainer = lime_image.LimeImageExplainer()
done = [2,1,3,4,5,6,7,8,9]
for i in range(len(X_train)):
    if len(done) == 10:
        break
    if Y_train[i] in done:
        continue

    done.append(Y_train[i])
    
    
    explanation = explainer.explain_instance(convert_vec(X_train[i]), classifier_fn = predict_fn, top_labels=10, hide_color=0, num_samples=50000, qs_kernel_size = 1)



    temp, mask = explanation.get_image_and_mask(Y_train[i], positive_only=True, num_features=784, hide_rest=False, min_weight = 0.01)
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
    plt.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    #plt.set_title('Positive Regions for {}'.format(Y_train[i]))
    #temp, mask = explanation.get_image_and_mask(Y_train[0], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
    #ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
    #ax2.set_title('Positive/Negative Regions for {}'.format(Y_train[i]))

    plt.show()
#plt.imshow(mask)
