import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

def WaldTest(x1, x2):
    n1, n2 = len(x1), len(x2)
    Mu1, Mu2 = np.mean(x1), np.mean(x2)
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    Theta = Mu2 - Mu1
    SE = np.sqrt(std1**2/n1 + std2**2/n2)
    W = Theta/SE
    Pvalue = chi2.sf(W**2, df=1)
    return Pvalue

alpha = 0.5
n = 20
x1 = np.random.poisson(lam=1, size=n)
x2 = np.random.poisson(lam=alpha, size=n)

mu = 1.5
nSIM = 1000
pval_all = np.zeros(nSIM)
for i in range(nSIM):
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(mu, 1, size=n)
    pval_all[i] = WaldTest(x1, x2)

print("Power:", np.mean(pval_all < 0.05 / 1000), "\n")

plt.hist(pval_all, bins=10)
plt.xlabel('pval_all')
plt.ylabel('Frequency');