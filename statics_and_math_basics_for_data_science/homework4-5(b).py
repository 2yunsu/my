import numpy as np
import math

n = 10

mu = 5
confidence = 0.05
eps = math.exp(mu)


data = np.random.randn(1,10)
T_Median = 0

B = 100
B_Median = np.zeros(B)

for i in range(B):
    B_sample = data
    B_Median[i] = np.median(B_sample)

# sns.hisplot(x=B_Median, bins=15)

def Get_CI(T, B_Sample):
    # Percentile Interval
    CI_Percent = list(np.quantile(B_Sample, q=[0.05, 0.95]))
    # Normal Interval
    se_b = np.std(B_Sample, ddof=1)
    CI_Normal = [T - 1.96*se_b, T + 1.96*se_b]
    # Pivot Interval
    CI_Pivot = [2*T - CI_Percent[1], 2*T - CI_Percent[0]]

    re = {'CI_Percent(5%, 95%)':CI_Percent, 'CI_Normal(5%, 95%)':CI_Normal, 'CI_Pivot(5%, 95%)':CI_Pivot}
    return re

print(Get_CI(T_Median, B_Median))