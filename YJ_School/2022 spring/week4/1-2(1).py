#1-2
import random
list=[]
while len(list)<6:
    i=random.randint(1,45)
    if i not in list:
        list.append(i)
print(list)