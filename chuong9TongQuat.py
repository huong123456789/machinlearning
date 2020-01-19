from __future__ import print_function 
import numpy as np 
from time import time # for comparing runing time

d, N = 1000, 10000 # kich thuoc-dac trung, va so diem du lieu 
X = np.random.randn(N, d) # N diem du lieu co so dac trung la d
M = 100 
Z = np.random.randn(M, d)

#tinh binh phuong khoang cach gia z va moi hang cua X tinh nhanh
def dist_ps_fast(z,X):
    X2=np.sum(X*X,1) #tinh tong binh phuong theo hang
    z2=np.sum(z*z)
    return X2 +z2-2*X.dot(z)
# tinh binh phuong khoan cach moi diem du lieu trong tap Z toi tap X 
def dist_ss_0(Z, X):
    M = Z.shape[0] 
    N = X.shape[0] 
    res = np.zeros((M, N)) 
    for i in range(M): 
        res[i] = dist_ps_fast(Z[i], X) 
    return res
# tinh binh phuong khoang cach cac diem du lieu trong tap X voi Z 
def dist_ss_fast(Z, X): 
    X2 = np.sum(X*X, 1) 
    Z2 = np.sum(Z*Z, 1) 
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)
    

t1 = time() 
D3 = dist_ss_0(Z, X) 
print('thoi gian tinh lan luot:', time() - t1, 's')

t1 = time() 
D4 = dist_ss_fast(Z, X) 
print('thoi gian tinh nhanh:', time() - t1, 's') 
print('Result difference:', np.linalg.norm(D3 - D4))
