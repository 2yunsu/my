list1=[]
list2=[]
for i in range(100):
    list1.append(0)

for j in range(100):
    list2.append(list1)

print(list2)