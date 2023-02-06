#숙제1  : 1~1000까지의 수 중 3의 배수만을 print 하시오
#while을 이용한 방법
num=1
while num<1000:
    if num%3==0:
        print("%d"% num)
    num=num+1

#for을 이용한 방법
for i in range(1,1001):
    if i%3==0:
        print("%d"%i)