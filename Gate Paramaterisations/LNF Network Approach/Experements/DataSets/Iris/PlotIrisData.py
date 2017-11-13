import numpy as np
import IrisData
import matplotlib.pyplot as plt

data, raw_targets = IrisData.read_data()

print(raw_targets)

class1 = []
class2 = []
class3 = []

for i in range(len(data)):
    if raw_targets[i] == 1:
        class1.append(np.array(data[i]))
    if raw_targets[i] == 2:
        class2.append(np.array(data[i]))
    if raw_targets[i] == 3:
        class3.append(np.array(data[i]))



class1 = np.array(class1)
class2 = np.array(class2)
class3 = np.array(class3)

print(class1)

plt.scatter(class1[:,2], class1[:,3], label='Setosa')
plt.scatter(class2[:,2], class2[:,3], label='Versicolour')
plt.scatter(class3[:,2], class3[:,3], label='Virginica')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()
