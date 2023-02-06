import numpy as np
import pandas as pd

df = pd.read_csv('coris.txt', sep=',', skiprows=[0, 1])

df0 = df[df['chd'] == 0]
df1 = df[df['chd'] == 1]

T_MeanDiff = np.mean(df1['ldl']) - np.mean(df0['ldl'])
print(T_MeanDiff)