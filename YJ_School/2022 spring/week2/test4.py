#Q1
def is_odd(a):
 if a%2==0: return "even"
 else: return "odd"

b=is_odd(2)
print(b)

#Q2
def aver(*args):
 result=0
 for i in args:
  result=result+i
 result=result/len(args)
 return result

#Q3
input1=input("첫번째 숫자를 입력하세요:")
input2=input("두번째 숫자를 입력하세요:")
int1=int(input1)
int2=int(input2)
total=int1+int2
print("두 수의 합은 %s입니다."%total)

#Q4
#3. ","는 띄어쓰기로 간주된다.
print("you" "need" "python")
print("you"+"need"+"python")
print("you", "need", "python")
print("".join(["you", "need", "python"]))

#Q5
f1=open("test.txt",'w')
f1.write("Life is too short")
f1.close()

f2=open("test.txt",'r')
print(f2.read())
f2.close()

#Q6
f=open("test.txt",'a')
a=input()
f.write(a)
f.write("\n")###
f.close()

#Q7
f1=open("test.txt",'r')
a=f1.read()
b=a.replace("java","python")
f1.close()

f2=open("test.txt",'w')
f2.write(b)
f.close()