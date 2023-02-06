import random
import matplotlib.pyplot as plt
list=[]
for i in range(100):
    a=random.uniform(-1,1)
    list.append(a)

plt.plot(sorted(list))
plt.show()