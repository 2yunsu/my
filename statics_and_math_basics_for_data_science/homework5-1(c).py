import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./homework5-1(c).csv")

T_Median = np.median(data)
print("Median:", T_Median, "\n")

# Bootstrap

B = 1000
B_std = np.zeros(B)

for i in range(B):
    B_sample = data.sample(frac=1, replace=True)
    B_std[i] = np.std(B_sample)

std_Mean = np.mean(B_std)
print(std_Mean)