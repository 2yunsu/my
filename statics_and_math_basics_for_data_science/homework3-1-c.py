import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

nRandom = 1000
n = 1000
p = 0.3
print(np.random.binomial(n, p, size=nRandom))
#
# nRandom = 1000
# Lambda = 10
#
# out = np.random.poisson(Lambda, nRandom)
# print('mean:', np.mean(out),"\n")
# print('var:', np.var(out), "\n")
#
# plt.hist(out)
# plt.xlabel('out')
# plt.ylabel('Frequency')
# plt.title('Histogram of out')
# plt.show()