import numpy as np
import sys

#2.4 c
x = np.array([[1,1,0], [0,1,1], [1,0,1]])
y = np.array([[1,2,3], [4,5,6], [7,8,9]])

res = np.dot(x,y)
print("2.4 c \n", res)
print()

#2.4 d
x = np.array([[1, 2, 1, 2], [4, 1, -1, -4]])
y = np.array([[0, 3], [1, -1], [2, 1], [5, 2]])

res = np.dot(x,y)
print("2.4 d \n", res)
print()

#2.4 e
x = np.array([[0, 3], [1, -1], [2, 1], [5, 2]])
y = np.array([[1, 2, 1, 2], [4, 1, -1, -4]])

res = np.dot(x,y)
print("2.4 e \n", res)
print()

#2.5 b
A = np.array([[1, -1, 0, 0, 1], [1, 1, 0, -3, 0], [2, -1, 0, 1, -1], [-1, 2, 0, -2, -1]])
b = np.array([[3], [6], [5], [-1]])

try:
	ATA = np.dot(A.T, A)
	ATA_inv = np.linalg.inv(ATA)
	ATb = np.dot(A.T, b)
	x = np.dot(ATA_inv, ATb)
	print("2.5 b", x)
	print()
except np.linalg.LinAlgError:
	print("2.5 b")
	print("there dose not exist inverse matrix of A.T*A")
	print()

#2.8 a
A = np.array([[2,3,4], [3,4,5], [4,5,6]])
try:
	A_inv = np.linalg.inv(A)
	print("2.8 a")
	print(A_inv)
	print()
except np.linalg.LinAlgError:
	print("2.8 a")
	print("there does not exist inverse matrix of A")
	print()

#2.10 a
x1 = np.array([[2],[-1],[3]])
x2 = np.array([[1],[1],[-2]])
x3 = np.array([[3],[-3],[8]])

x = np.hstack([x1,x2,x3])
rank = np.linalg.matrix_rank(x)

print("2.10 a")
print("rank:", rank)
print("len:", len(x))
if len(x) == rank:
	print("linear independent")

else:
	print("linear dependent")
print()

#2.10 b
x1 = np.array([[1], [2], [1], [0], [0]])
x2 = np.array([[1], [1], [0], [1], [1]])
x3 = np.array([[1], [0], [0], [1], [1]])

x = np.hstack([x1,x2,x3])
rank = np.linalg.matrix_rank(x)

print("2.10 b")
print("rank:", rank)
print("len:", len(x))
if len(x) == rank:
	print("linear independent")

else:
	print("linear dependent")
print()

#2.11
x1 = np.array([[1], [1], [1]])
x2 = np.array([[1], [2], [3]])
x3 = np.array([[2], [-1], [1]])
y = np.array([[1], [-2], [5]])

x = np.hstack([x1,x2,x3])
x_inv = np.linalg.inv(x)
print("x",x)
print("xinv:", x_inv)
lam = np.dot(x_inv, y)
print("2.11")
print(lam)
print()

#2.12
