from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
#chieu cao (cm), dau vao du lieu, moi hang la mot diem du lieu
X=np.array([[147,150,153,158,163,165,168,170,173,175,178,180,183]]).T
#can nang tuong ung(kg)
Y=np.array([49,50,51,54,58,59,60,62,63,64,66,67,68])
#xay dung mang X bias
one= np.ones((X.shape[0],1))#cung cap ma tran co chieu la x.shape[0]*1
#moi diem du lieu la mot hang co dang [w0,w1]
Xbias= np.concatenate((one,X),axis=1)
#tinh toan can nang voi moi dong tuong ung
A= np.dot(Xbias.T,Xbias)
b= np.dot(Xbias.T,Y)
w= np.dot(np.linalg.pinv(A),b)# lay ma tran gia ghich dao cua A
w_0, w_1 =w[0], w[1]

y1= w_1*155+w_0
y2= w_1*160+w_0
print('155 cm: can nang thuc la 52Kg, can nang tinh duoc la %.2fkg' %(y1))
print('160 cm: can nang thuc la 56Kg, can nang tinh duoc la %.2fkg' %(y2))

#giai theo thu vien scikit-learn de tim nghiem
# fit the model by Linear Regression
regr= linear_model.LinearRegression()
regr.fit(X,Y)# in scikit-learn, each sample is one row
#so sanh ket qua
print("cach thong thuong   :w_1=",w_1,"w_0=",w_0)
print("scikit-learn'solution:w_1=",regr.coef_[0], "w_0=",regr.intercept_)
