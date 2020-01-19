from __future__ import print_function 
import numpy as np 
import matplotlib.pyplot as plt #dung de ve do thi
from scipy.spatial.distance import cdist 
import random 
np.random.seed(15)
means = [[2, 2], [8, 3], [3, 6]] #ki vong duoc tao
cov = [[1, 0], [0, 1]] # ma tran hiep phuong sai
N = 500 
#tao du lieu theo phan phoi chuan cho bai toan
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis = 0) #gop du lieu lai
K = 3 # 3 clusters 
#cung cap mang cac lable ung voi 1500 diem du lieu
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

#ham khoi toa cac centroids ban dau:
def kmeans_init_centroids(X,k):
    return X[np.random.choice(X.shape[0], k, replace="False" )]

#ham de tim lable moi cho cac diem khi co dinh ca centroid
#luu y lable o day mang gia tri 0,1,2
def  kmeans_assign_labels(X, centroids):
    D = cdist(X, centroids) #tin khoang cach 
    return np.argmin(D, axis = 1)#tra ve chi so K cua centroids gan nhat 

#so sanh 2 centroids, kiem tra dieu kien dung cua thuat toan
def has_converged(centroids, new_centroids): 
    # neu giong nhau thi tra ve true
    return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids]))

#cap nhat cac centroids khi biet label cua moi diem du lieu
def kmeans_update_centroids(X, labels, K): 
    centroids = np.zeros((K, X.shape[1])) 
    for k in range(K):
       #xet cac diem du lieu (theo hang) duoc gan trong cum k 
        Xk = X[labels == k, :] 
        #lay trung binh cong tho cong thc 10.7
        centroids[k,:] = np.mean(Xk, axis = 0)
    return centroids

#ham main
def kmeans(X,K):
    centroids = [kmeans_init_centroids(X,K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X,centroids[-1]))
        new_centroids=kmeans_update_centroids(X,labels[-1],K)
        if has_converged(centroids[-1],new_centroids):
            break
        centroids.append(new_centroids)
        it+=1
        return (centroids, labels, it)   
        
(centroids, labels, it) = kmeans(X, K) 
print('Centers found by our algorithm:\n', centroids[-1]) 
print (labels[-1].shape)
