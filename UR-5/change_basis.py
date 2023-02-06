import numpy as np


arr=np.array([[1,3,5],[5,6,7],[1,1,1]])
print(arr)
print(arr.shape)
print(arr.ndim)
arr_inv=np.linalg.inv(arr)
print(arr_inv)
a=np.array([[[1,2,3]],[[4,5,6]],[[7,8,9]]])
print(a.shape)
inv=np.linalg.inv(a)