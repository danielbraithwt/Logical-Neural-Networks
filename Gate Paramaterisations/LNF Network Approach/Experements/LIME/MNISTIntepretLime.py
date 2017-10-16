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

model = np.load('network.npy')

def softmax(vals):
    ex = np.exp(vals)
    return ex/np.sum(ex)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def apply_model(m, img):

    prev_out = img
    for l in range(len(m)):
        layer = m[l]
        #print(layer)
        weights = layer[0]
        biases = layer[1]

        out = np.add(np.dot(prev_out, np.transpose(weights)), biases)
        if not l == len(m)-1:
            out = sigmoid(out)
        prev_out = out

    #print(prev_out)
    return softmax(prev_out)[0]
        

        

def predict_fn(imgs):
    predictions = []
    for img in imgs:
        #print(img.shape)
        
        img = rgb2gray(img)
        #print(img.shape)
        img = np.reshape(img, (-1))
        predictions.append(apply_model(model, img))

    #print(predictions)
    return predictions
    


def convert_vec(img):
    img = np.reshape(img, (28, 28))
    img = gray2rgb(img)
    #print(img)
    return img


#print(predict_fn([convert_vec(X_train[0])]))


explainer = lime_image.LimeImageExplainer()
done = [0,2,3,4,5,6,7,8,9]
for i in range(len(X_train)):
    if len(done) == 10:
        break
    if Y_train[i] in done:
        continue

    done.append(Y_train[i])
    
    
    explanation = explainer.explain_instance(convert_vec(X_train[i]), classifier_fn = predict_fn, top_labels=10, hide_color=0, num_samples=50000, qs_kernel_size = 1)



    temp, mask = explanation.get_image_and_mask(Y_train[i], positive_only=True, num_features=784, hide_rest=False, min_weight = 0.0005)
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
    plt.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    #plt.set_title('Positive Regions for {}'.format(Y_train[i]))
    #temp, mask = explanation.get_image_and_mask(Y_train[0], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
    #ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
    #ax2.set_title('Positive/Negative Regions for {}'.format(Y_train[i]))

    plt.show()
