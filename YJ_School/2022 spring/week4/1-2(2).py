#1-2(다른 방법)
import random
def fortyfive():
	data=[]
	for i in range(1,45):
		data.append(i)
	return(data)

a=fortyfive()

random.shuffle(a)

for i in range(6):
print(a.pop())