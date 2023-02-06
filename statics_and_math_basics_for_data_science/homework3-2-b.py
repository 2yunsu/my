import numpy as np
import statistics

nRandom = 20
n = 1
p = 0.5
c_list = []

#1000번 반복
for j in range(1000):
    c = 20000
    for i in range(20):
        z = np.random.binomial(n, p, size=nRandom)
        if z[i] == 0:
            c = c*2
        else:
            c = c/2
    c_list.append(c)

c_mean = statistics.mean(c_list)
print("c_mean:", c_mean)

#주어진 공식으로 계산한 값
cal_value = ((5/4)**20)*20000
print("cal_value", cal_value)