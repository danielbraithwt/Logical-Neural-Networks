import numpy as np
test = 'SPECTF.test'
train = 'SPECTF.train'

def read_file(name):
    lines = []
    f = open(name, 'r+')
    for line in f:
        lines.append(line)
    f.close()

    examples = []
    targets = []

    for line in lines:
        splt = line.split(',')
        target = int(splt[0])
        example = [int(splt[x]) for x in range(1, len(splt))]

        targets.append([target])
        examples.append(example)

    return np.array(examples), np.array(targets)

def normalise(examples):
    return examples * 1.0/100.0

def read_data():
    X_test, Y_test = read_file(test)
    X_train, Y_train = read_file(train)

    X_test = normalise(X_test)
    X_train = normalise(X_train)

    return (X_train, Y_train), (X_test, Y_test)
