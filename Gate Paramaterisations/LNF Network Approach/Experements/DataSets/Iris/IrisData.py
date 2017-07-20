import numpy as np

def read_data():
    instances = []
    targets = []

    f = open('iris.data', 'r+')

    f1 = []
    f2 = []
    f3 = []
    f4 = []
    
    for line in f:
        s = line.strip().split(',')
        c = s[-1]
        clas = None
        if c == 'Iris-setosa':
            clas = 1
        elif c == 'Iris-versicolor':
            clas = 2
        elif c == 'Iris-virginica':
            clas = 3
        o_instance = [float(x) for x in s[0:-1]]

        f1.append(o_instance[0])
        f2.append(o_instance[1])
        f3.append(o_instance[2])
        f4.append(o_instance[3])
        targets.append(clas)

    f1 = np.expand_dims(_normalize(f1), 1)
    f2 = np.expand_dims(_normalize(f2), 1)
    f3 = np.expand_dims(_normalize(f3), 1)
    f4 = np.expand_dims(_normalize(f4), 1)

    instances = np.concatenate((f1,f2,f3,f4), axis=1)

    return instances, targets

def read_data_raw():
    instances = []
    targets = []

    f = open('iris.data', 'r+')

    f1 = []
    f2 = []
    f3 = []
    f4 = []
    
    for line in f:
        s = line.strip().split(',')
        c = s[-1]
        clas = None
        if c == 'Iris-setosa':
            clas = 1
        elif c == 'Iris-versicolor':
            clas = 2
        elif c == 'Iris-virginica':
            clas = 3
        o_instance = [float(x) for x in s[0:-1]]

        instances.append(o_instance)
        targets.append(clas)

    return np.array(instances), targets

def _normalize(feature):
    feature = np.array(feature)
    return (feature - feature.min())/(feature.max() - feature.min())


read_data()
