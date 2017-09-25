#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tensorflow.examples.tutorials.mnist import input_data

dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def transform(weights):
  return np.log(1 + np.exp(-weights))


def inv_transform(weights):
  return -np.log(np.exp(weights)-1)

def gen_weights(shape):
    var = np.sqrt(np.log(4 * (4 + 3*shape[1])))
    mean = -(1.0/2.0) * np.log(shape[1]**2 * (4 + 3*shape[1]))

    #w = np.random.exponential(2.31, shape) * (1.0/shape[1])
    
    #print(transform(w))
    
    initial = np.random.lognormal(mean, var, shape)
    #initial = np.random.uniform(-0.54132, 7, shape)#
    #initial = stats.betaprime.rvs((14.0/(3.0 * shape[1])), (10.0/3.0), size=shape) + 1e-15
    #w = transform(initial) * 1.0/shape[1]
    #w = inv_transform(w)
    #w = inv_transform(initial)
    #print(transform(w))
    #w = inv_transform(np.abs(np.random.normal(0, 1.0/shape[1], shape)))

    #beta = (4.0 + 8.0 * shape[1])/(3.0 * shape[1])
    #alpha = (10.0 * shape[1] + 8.0)/(3.0 * shape[1] * shape[1])

    #initial = stats.betaprime.rvs(alpha, beta, size=shape) + 1e-20
    #initial = np.random.poisson(2.0/shape[1], shape) + 1e-15
    print("I")
    w = inv_transform(initial)
    print("T")
    print(transform(w))

    return w

def construct_network(num_inputs, hidden_layers, num_outputs):
  network = []
    
  layers = hidden_layers
  layers.append(num_outputs)

  layer_ins = num_inputs
  for l in layers:
    weights = tf.Variable(gen_weights((l, 2 * layer_ins)), dtype='float32')
    bias = tf.Variable(np.zeros((1, l)) + 27.6309, dtype='float32')

    network.append([weights, bias])
    layer_ins = l

  return network

def transform_weights(weights):
  return tf.log((1 + tf.exp(-weights)))

def noisy_or_activation(inputs, weights, bias):
  t_w = transform_weights(weights)
  t_b = transform_weights(bias)

  z = tf.add(tf.matmul(inputs, tf.transpose(t_w)), t_b)
  return 1 - tf.exp(-z)

def noisy_and_activation(inputs, weights, bias):
  t_w = transform_weights(weights)
  t_b = transform_weights(bias)

  z = tf.add(tf.matmul(1 - inputs, tf.transpose(t_w)), t_b)
  return tf.exp(-z)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.sigmoid)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Logical Neural Network
  activations = [noisy_or_activation, noisy_or_activation, noisy_or_activation, noisy_and_activation]
  lnn = construct_network(7 * 7 * 64, [1024, 512, 256], 10)

  prev_out = pool2_flat
  for idx in range(len(lnn)):
    prev_out = tf.concat([prev_out, 1 - prev_out], axis=1)
        
    layer = lnn[idx]
    act = activations[idx]
        
    w = layer[0]
    b = layer[1]

    out = act(prev_out, w, b)
    prev_out = out

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = prev_out
  preds = logits * (1/tf.reduce_sum(logits))
  

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": preds
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  y = onehot_labels
  #loss = tf.losses.softmax_cross_entropy(
  #    onehot_labels=onehot_labels, logits=logits)

  y_hat_prime_0 = tf.clip_by_value(preds, 1e-20, 1)
  y_hat_prime_1 = tf.clip_by_value(1 - preds, 1e-20, 1)
  errors = -(y * tf.log(y_hat_prime_0) + (1-y) * tf.log(y_hat_prime_1))#
  error = tf.reduce_sum(errors)

  loss = error

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = input_data.read_data_sets(dir_path + "/")#tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  #mnist_classifier = tf.estimator.Estimator(
  #    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": preds}
  #logging_hook = tf.train.LoggingTensorHook(
  #    tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1,
      num_epochs=None,
      shuffle=False)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=len(train_data) * 30,)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
