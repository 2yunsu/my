import numpy as np
from scipy.stats import chi2

def WaldTest(x1, x2):
    n1, n2 = len(x1), len(x2)
    lambda1, lambda2 = np.mean(x1), np.mean(x2)
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    Theta = lambda2 - lambda1
    SE = np.sqrt(std1**2/n1 + std2**2/n2)
    W = np.sqrt(n)*(Theta/SE)
    Pvalue = chi2.sf(-abs(W), df=1)
    return Pvalue

alpha = 0.5
n = 20
x1 = np.random.poisson(lam=1, size=n)
x2 = np.random.poisson(lam=alpha, size=n)
print(WaldTest(x1, x2))