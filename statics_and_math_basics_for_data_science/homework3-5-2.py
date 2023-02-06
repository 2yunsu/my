import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

#a
n_case_trajectory_df = pd.DataFrame(columns=range(11))

for _ in range(100):
    n_current_case = 5
    n_case_trajectory = []
    n_case_trajectory.append(5)
    for i in range(10):
        n_contact = np.random.poisson(10, n_current_case)
        n_new_case = np.random.binomial(n_contact, 0.2)
        n_current_case = np.sum(n_new_case)
        n_case_trajectory.append(n_current_case)
    n_case_trajectory_df = n_case_trajectory_df.append\
        (pd.Series(n_case_trajectory, index=n_case_trajectory_df.columns),\
         ignore_index=True)

plt.boxplot(n_case_trajectory_df)
plt.title('before 50days')
plt.xlabel('time unit')
plt.ylabel('infected individuals');
# plt.savefig('./homework3-5-a.png')
plt.show()
plt.close()

#b

n_case_trajectory_new_df = pd.DataFrame(columns=range(2))

for _ in range(100):
    n_case_trajectory_new = []
    # n_case_trajectory_new.append(n_current_case)
    for i in range(2):
        n_contact = np.random.poisson(10, n_current_case)
        n_new_case = np.random.binomial(n_contact, 0.14)
        n_current_case_2 = np.sum(n_new_case)
        n_case_trajectory_new.append(n_current_case_2)
    n_case_trajectory_new_df = n_case_trajectory_new_df.append\
        (pd.Series(n_case_trajectory_new, index=n_case_trajectory_new_df.columns),\
         ignore_index=True)

plt.boxplot(n_case_trajectory_new_df)
plt.title('after 50days')
plt.xlabel('time unit')
plt.ylabel('infected individuals');
# plt.savefig('./homework3-5-b.png')
plt.show()
plt.close()

#c
n_case_trajectory_c_df = pd.DataFrame(columns=range(8))

for _ in range(100):
    n_case_trajectory_c = []
    # n_case_trajectory_c.append(n_current_case_2)
    for i in range(8):
        n_contact = np.random.poisson(3, n_current_case_2)
        n_new_case = np.random.binomial(n_contact, 0.14)
        n_current_case_3 = np.sum(n_new_case)
        n_case_trajectory_c.append(n_current_case_3)
    n_case_trajectory_c_df = n_case_trajectory_c_df.append\
        (pd.Series(n_case_trajectory_c, index=n_case_trajectory_c_df.columns),\
         ignore_index=True)

plt.boxplot(n_case_trajectory_c_df)
plt.title('after 60days')
plt.xlabel('time unit')
plt.ylabel('infected individuals');
# plt.savefig('./homework3-5-final.png')
plt.show()
plt.close()

#a,b,c 그래프 합치기
final_df = pd.concat([n_case_trajectory_df, n_case_trajectory_new_df, n_case_trajectory_c_df], axis=1)

plt.boxplot(final_df)
plt.title('total')
plt.xlabel('time unit')
plt.ylabel('infected individuals');
# plt.savefig('./homework3-5-final.png')
plt.show()
plt.close()