import numpy as np
from PIL import Image
import glob

TRAINING_FILE = "svm.train.normgrey"
TEST_FILE = "svm.test.normgrey"

def to_one_hot(val, m):
    if val == -1:
        val = 0
    
    vec = np.zeros(m)
    vec[val] = 1
    return vec

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def read_images(name):
    examples = []
    targets = []
    
##    for filename in glob.glob(name + '/face/*.pgm'):
##        img = PIL2array(Image.open(filename)).reshape(-1)
##        img = img/255
##        examples.append(img)
##        targets.append([0,1])
##
##        num_faces += 1
##
##    for filename in glob.glob(name + '/non-face/*.pgm'):
##        img = PIL2array(Image.open(filename)).reshape(-1)
##        img = img/255
##        examples.append(img)
##        targets.append([1,0])
##
##        num_non_faces += 1
##
##        if num_faces == num_non_faces:
##            break
    raw_instances = []
    f = open(name, 'r+')
    f.readline()
    f.readline()
    for line in f:
        raw_instances.append(line.strip().split(' '))
    f.close()

    print(len(raw_instances))

    examples = []
    targets = []

    for instance in raw_instances:
        parsed = [float(x) for x in instance]
        examples.append(parsed[0:-2])
        targets.append(to_one_hot(int(parsed[-1]), 2))
##
    return examples, targets



def read_data():
    return read_images(TRAINING_FILE), read_images(TEST_FILE)


if __name__ == "__main__":
    read_data()
