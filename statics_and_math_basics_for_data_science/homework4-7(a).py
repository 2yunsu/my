import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('coris.txt', sep=',', skiprows=[0, 1])

sns.scatterplot(x=df['tobacco'], y=df['sbp'])
T_Corr = np.corrcoef(df['tobacco'], df['sbp'])[0, 1]

n = df.shape[0]
B = 1000
B_Corr = np.zeros(B)

for i in range(B):
    B_sample_idx = np.random.choice(n, size=n, replace=True)
    df1 = df.iloc[B_sample_idx, ]
    B_Corr[i] = np.corrcoef(df1['tobacco'], df1['sbp'])[0, 1]

sns.histplot(B_Corr, bins=15)