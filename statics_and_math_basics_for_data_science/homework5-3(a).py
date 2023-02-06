import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 100
x = np.random.exponential(scale=5, size=n)

muhat = 1/np.mean(x)
logmu = math.log(muhat)

nSIM = 1000

def ParaBoot():
    PB = np.zeros(nSIM)
    for i in range(nSIM):
        X1 = np.random.exponential(scale=5/muhat, size=n)
        muhat1 = 1/np.mean(X1)
        logmuhat1 = math.log(muhat1)
        PB[i] = logmuhat1

    return PB

def Boot():
    B_std = np.zeros(nSIM)
    for i in range(nSIM):
        B_sample = np.random.exponential(scale=5/muhat, size=n)
        B_std[i] = np.mean(B_sample)

    return B_std



print('From Deltamethod: ', 1/math.sqrt(n))
print('From Bootstrap: ', np.std(Boot()))
print('From Parametric Bootstrap: ', np.std(ParaBoot()), "\n")


x = np.arange(0, 20, 0.001)
plt.figure(figsize=(15,10))
plt.title('From Deltamethod')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.plot(x, norm.pdf(x, loc=10, scale=2))
plt.show()

sns.histplot(x=Boot(), bins=15)
plt.title('From Bootstrap')
plt.show()

sns.histplot(x=ParaBoot(), bins=15)
plt.title('From Parametric Bootstrap')
plt.show()

# x_axis = np.arange( -10, 10, 0.001 )
# plt.plot(x_axis, ParaBoot())


