import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats

n = 500
res = []
iters = 100000


def __perms(n):
    if not n:
        return

    p = []

    for i in range(0, 2**n):
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p

def __n_rand_perms(n, size):
    if not n:
        return

    idx = [random.randrange(2**n) for i in range(size)]

    p = []

    for i in idx:
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p


def generateExpressions(n):
    inputs = __perms(n)
    outputs = __n_rand_perms(len(inputs), 1)

    return np.array(list(map(lambda x: (inputs, x), outputs)))

#var = np.sqrt(np.log(4 * (4 + 3*n)))
#mean = -(1.0/2.0) * np.log(n*n * (4 + 3*n))

def inv_transform(weights):
    return -np.log(np.exp(weights)-1)

def transform_weights(weights):
    return np.log((1 + np.exp(-weights)))


#expression = generateExpressions(n)[0]
#data = expression[0]


#w = np.random.lognormal(mean, var, n)
#w = np.random.poisson((2.0/n), n)
#w = transform_weights(inv_transform(w))

var = np.sqrt(np.log(4 * (4 + 3*n)))
mean = -(1.0/2.0) * np.log(n**2 * (4 + 3*n))

iters = 100

means = []
v = []

for i in range(30):
    res = []

    #print(n)
    #print(mean)
    #print(var)
    w = np.random.lognormal(mean, var, (n))
    for i in range(iters):
        #w = stats.betaprime.rvs((14.0/(3.0 * n)), (10.0/3.0), size=n)
        x = np.random.randint(0, 2, n)
        z = np.sum(np.multiply(x, w))
        y = 1 - np.exp(-z)

        res.append(y)

    res = np.array(res)
    means.append(res.mean())
    v.append(res.var())


means = np.array(means)
v = np.array(v)
print(means.mean())
print(v.mean())

#x = np.linspace(stats.betaprime.ppf(0.01, (14.0/(3.0 * n)), (10.0/3.0)), stats.betaprime.ppf(0.99, (14.0/(3.0 * n)), (10.0/3.0)), 100)
#rv = stats.betaprime((14.0/(3.0 * n)), (10.0/3.0))
#plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#plt.show()
