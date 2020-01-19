from __future__ import print_function
import numpy as np 
from time import time # dung de lay thoi gian chay, so sanh thoi gian
d, N = 1000, 10000 # kich thuoc-dac trung, va so diem du lieu 
X = np.random.randn(N, d) # N diem du lieu co so dac trung la d
z = np.random.randn(d)

#tinh binh phuong khoan cach giua z va x
def dist_pp(z, x):
     d = z - x.reshape(z.shape)#dua x va cung dang voi z roi lay hieu
     return np.sum(d*d)#tinh tong binh phuong hieu cac khoang cach

#tinh binh phuong khoan cach giua z va moi hang cua X
def dist_ps_naive(z,X):
    N = X.shape[0]#lay so diem du lieu
    res = np.zeros((1, N))#luu ket qua cua khoang cach N diem du lieu do
    for i in range(N):
         res[0][i] = dist_pp(z, X[i])
    return res

#tinh binh phuong khoang cach z va moi hang cua X
# theo cong thuc tinh nhanh
def dist_ps_fast(z,X):
    X2=np.sum(X*X,1) #tinh tong binh phuong theo hang
    z2=np.sum(z*z)
    return X2 +z2-2*X.dot(z)
#in ra ket qua
t1 = time()
D1 = dist_ps_naive(z, X) 
print('thoi gian cach tinh lan luot:', time() - t1, 's')

t1 = time() 
D2 = dist_ps_fast(z, X) 
print('thoi gian khi tinh nhanh:', time() - t1, 's') 
print('chenh lech ket qua:', np.linalg.norm(D1 - D2))
