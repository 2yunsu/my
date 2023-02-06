#숙제2 : 1~1000까지의 수 중 3의 배수의 총 합을 구하시오
#while을 사용한 방법
i=1
sum=0
while i<1000:
    if i%3==0:
        sum=sum+i
    i=i+1
print(sum)

#for을 사용한 방법
sum=0
for i in range(1000):
    if i%3==0:
        sum=sum+i
    i=i+1
print(sum)