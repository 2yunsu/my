#숙제3: 위키독스 3장 연습문제
#Q1
#답: "shirt"
#이유:
"""
1) 3번째 줄의 'wife'는 a에 없으므로 넘어감
2) 'python'은 있으나 'you' 또한 있으므로 넘어감
3) 'shirt'가 a에 없으므로 shirt 출력
4) 'need'가 a에 있으나 3에서 출력됐으므로 미출력
5) 3에서 출력됐으므로 미출력
"""
a= "Life is too short, you need python"

if "wife" in a: print("wife")
elif "python" in a and "you" not in a: print("python")
elif "shirt" not in a: print("shirt")
elif "need" in a: print("need")
else: print("none")

#Q2
i=1
sum=0
while i<1000:
    if i%3==0:
        sum=sum+i
    i=i+1
print(sum)

#Q3
i=1
while i<=5:
    print("*"*i)
    i=i+1

#Q4
for i in range(1,101):
    print(i)

#Q5
a=[70,60,55,75,95,90,80,80,85,100]
sum=0
for i in a:
    sum=sum+i
print(sum//len(a))

#Q6
a=[1,2,3,4,5]
result=[num*2 for num in a if num%2==1]
print(result)