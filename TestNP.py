import numpy as np

a=np.random.randn(3,2)
print(a)
b= 5+2*a
print(b)
c = np.array([(1, 2, 3), (4, 5, 6)])
print(c.T)
x= np.arange(3)
print(x)
print(x+c)
x
print(np.sum(c,0))
m=.10
print(m)
k=np.random.randn(1,7)
print(k)
h=np.random.randn(6,1)
print(h)
l=h+k #kiem tra tinh mo rong trong ma tran
print(l)
print(l.shape)
def dist_ps_fast(z,X):
    X2=np.sum(X*X,1) #tinh tong binh phuong theo hang
    print(X2)
    z2=np.sum(z*z)
    print(z2)
    return X2 +z2-2*X.dot(z)
z=np.array([1,3])
print(z)
X1=np.array([(5,2),(1,4),(2,1)])
print(X1)
print(X1.dot(z.T))
print(dist_ps_fast(z,X1))
N=3
arr=np.asarray([0]*N + [1]*N + [2]*N).T
print("mang tim duoc la\n")
print(arr)
a1= np.random.randn(2,3)
a2= np.random.randn(4,3)
print(a1.dot(a2.T))
print(set(tuple(a3) for a3 in a1))
a4=np.array([(1,6,9),(5,9,2),(2,9,1)])
print(a4[-1])
print(np.argmin(a4))
def kmeans_init_centers(X, k):
    # con k hang bat ki tu x
    return X[np.random.choice(X.shape[0], k)]
    #  X[[1, 5, 10], :]
print(kmeans_init_centers(a4,2))
a5=np.array([1,5,5,9,8,2])
print(a5[a5==5,:])