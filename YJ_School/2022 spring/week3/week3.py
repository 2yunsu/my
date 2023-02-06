#Q1
class FourCal:
    def __init__(self,first,second):
        self.first=first
        self.second=second
    def setdata(self, first, second):
        self.first=first
        self.second=second
    def add(self):
        result=self.first+self.second
        return result
    def mul(self):
        result = self.first*self.second
        return result
    def sub(self):
        result = self.first-self.second
        return result
    def div(self):
        result = self.first/self.second
        return result

class MoreFourCal(FourCal):
    pass

a=MoreFourCal(4,2)
a.add()
print(a)
"""
사칙연산을 하는 FourCal 클래스와, 이를 상속하는 MoreFourCal 클래스가
있다고 할 때, 아래와 같이 하면 부모 클래스의 함수를 사용할 수 있다.
"""

#Q2
"""
if __name__=="__main__"는, 해당 파일을 직접 실행했을 때와
다른 파일에서 import를 통해 함수만을 사용할 때 다른 결과를 얻기 위해 사용된다.
직접 그 파일을 실행할 때는 if문이 참이되어 조건문을 실행하고,
다른 파일에서 import를 통해 파일을 실행하면 if문이 거짓이 되어
조건문이 실행되지 않는다.
"""

#Q3
"""
모듈: 모듈이란 하나의 파이썬 파일로, 함수, 변수, 클래스로 구성될 수 있으며
        import를 통해 불러올 수 있다.
패키지: 모듈의 상위 개념으로, 모듈을 디렉터리로 나누어 종합적으로 이용할 수 있다.
라이브러리: 패키지의 상위 개념으로 하나의 프로그램으로 특정 목적으로 이용된다.
            라이브러리는 상대적으로 개발자의 접근이 능동적이라는 점에서
            프레임워크와 대응되는 개념으로 쓰이기도 한다.
"""

#Q4
# try, except, pass를 이용하여 오류를 무시할 수 있다.
try:
    4/0
except ZeroDivisionError:
    pass

#Q5
#abs: 절댓값 찾기.
abs(-5)
#int: 정수형으로 변환, 혹은 10진수로 변환
int(3.5)
int('0o12',8)
int(0o12)
#len: 길이를 구해줌
len([1,2,3])
#max: 최댓값 구하기
max([1,2,3])
#min: 최솟값 구하기
min([1,2,3])
#round: 반올림
round(1.23213,3)
#str: 문자형으로 변환
s=str(3)
type(s)

#Q6
#sort와 sorted는 입력을 정렬하는 함수로, sorted는 그 결과를 리스트로
#돌려주지만 sort는 결과를 돌려주지 않는다.
a=[8,6,4,3,1,2]
sorted(a)
a.sort()
print(a)
a.sort(reverse=True) #내림차순 정렬

#Q7
import random
random.random()
random.randint(1,45)

#Q8
import time
start=time.time()
for i in range(1,1001):
    if i%3==0:
        print("%d"%i)
end=time.time()
print(end-start)