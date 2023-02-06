import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('coris.txt', sep=',', skiprows=[0, 1])

plt.subplot(1, 1, 1)
sns.boxplot(data=df, x='tobacco', y='ldl')
plt.xlabel('tobacco')
plt.ylabel('LDL')
plt.title('LDL by tobacco')
plt.show()