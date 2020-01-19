import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
import random
np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, 500)
X1 = np.random.multivariate_normal(means[1], cov, 500)
X2 = np.random.multivariate_normal(means[2], cov, 500)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3
#khoi tao cac centers ban dau
def kmeans_init_centers(X, k):
    # con k hang bat ki tu x
    return X[np.random.choice(X.shape[0], k)]
    #  X[[1, 5, 10], :]

#tim label moi khi centers co dinh
#cac phan tu cua lable nhan cac gia trị; 0,1,2
def kmeans_assign_labels(X, centers):
   #tinh khoan cach giua data va center
    D = cdist(X,centers)
    # tra ve chi so gan center nhat
    return np.argmin(D, axis = 1)
    
#cap nhat centrers khi biet label cua moi diem du lieu
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))#(3,2)
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

#kiem tra dieu kien dung
def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))
#ham goc

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    #print(centers[-1])
    labels = []
    max_it = 50
    it = 0 
    while it < max_it:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print ('center tim duoc la:\n', centers[-1])
print (labels[-1].shape)
#print (labels)
print(it)

#dung thu vien scikit-learn
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=0).fit(X) 
print('Centers found by scikit-learn:') 
print(model.cluster_centers_) 
pred_label = model.predict(X) 
print(pred_label.shape)
