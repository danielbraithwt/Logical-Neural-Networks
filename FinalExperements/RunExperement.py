import sys
import os
import MultiOutLNN
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import requests
import json
import gc
 
def send_notification_via_pushbullet(title, body):
    """ Sending notification via pushbullet.
        Args:
            title (str) : title of text.
            body (str) : Body of text.
    """
    data_send = {"type": "note", "title": title, "body": body}
 
    ACCESS_TOKEN = 'o.1fkgpH63KmfKEFwmhsnIWWFcyJQOseZG'
    resp = requests.post('https://api.pushbullet.com/v2/pushes', data=json.dumps(data_send),
                         headers={'Authorization': 'Bearer ' + ACCESS_TOKEN, 'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Something wrong')
    else:
        print('complete sending')


experement_file = sys.argv[1]
job_id = sys.argv[2]
task_id = sys.argv[3]

print("Loading MNIST...")
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + "/")
print(tf.gfile.Exists(dir_path + "/"))
mnist = input_data.read_data_sets(dir_path + "/", one_hot=True)

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

num_inputs = len(X_train[0])
num_outputs = len(Y_train[0])

#testing_examples = tf.constant(X_test)
#testing_labels = tf.constant(Y_test)

print("MNIST Loaded...")

print("Reading Experement: ", experement_file)

f = open(experement_file, 'r+')
name = f.readline().strip()
#local_mnist = f.readline().strip() == 'True'
num = int(f.readline().strip())
out = "{}-{}.txt".format(f.readline().strip(), task_id)
raw_layers = f.readline().strip()
raw_activations = f.readline().strip()
iterations = int(f.readline().strip())
lsm = f.readline().strip() == 'True'
add_not = f.readline().strip() == 'True'
f.close()

layers = [int(l) for l in raw_layers.split()]
activations = []
iterations = len(X_train) * iterations
for ra in raw_activations.split():
    if ra == 'AND':
        activations.append(MultiOutLNN.noisy_and_activation)
    elif ra == 'OR':
        activations.append(MultiOutLNN.noisy_or_activation)

print("\nConfig")
print("layers: ", layers)
print("activations: ", activations)
print("lsm: ", lsm)
print("not: ", add_not)

if not os.path.exists("./" + name):
    os.makedirs(name)

f = open("./{}/{}".format(name, out), 'w+')
f.close()

print()
print("Starting Experement")
#send_notification_via_pushbullet("Experement ({} : {}): {}".format(job_id, task_id, name), "Starting")
for i in range(num):
    gc.collect()
    print("Starting Iteration: ", i)
    with tf.Graph().as_default():
        res = MultiOutLNN.train_lnn(X_train, Y_train, iterations, num_inputs, layers, num_outputs, activations, add_not, lsm)

        training_wrong = MultiOutLNN.run_lnn(X_train, Y_train, res, num_inputs, layers, num_outputs, activations, add_not)
        testing_wrong = MultiOutLNN.run_lnn(X_test, Y_test, res, num_inputs, layers, num_outputs, activations, add_not)
    tf.reset_default_graph()

    training_error_rate = training_wrong/len(X_train)
    testing_error_rate = testing_wrong/len(X_test)

    print("Training Error Rate: {}%".format(training_error_rate))
    print("Testing Error Rate: {}%".format(testing_error_rate))

    f = open("./{}/{}".format(name, out), 'a')
    f.write("{} : {}\n".format(training_error_rate, testing_error_rate))
    f.close()
    
    del f
    del res
    gc.collect()


#f = open("./{}/{}".format(name, out), 'r+')
#send_notification_via_pushbullet("Experement Results ({} : {}): {}".format(job_id, task_id, name), str(f.readlines()))
#f.close()
