import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0,5,3001)
y = np.exp(-z)

plt.ylabel("$e^{-z}$")
plt.xlabel("z")
plt.plot(z, y)
plt.show()
