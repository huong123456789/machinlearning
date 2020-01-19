from __future__ import print_function
import numpy as np 
from sklearn import neighbors, datasets 
from sklearn.model_selection import train_test_split # for splitting data 
from sklearn.metrics import accuracy_score # for evaluating results

np.random.seed(7)#dam bao nhan duoc ket qua tuong tu khi chay lai
iris = datasets.load_iris() 
iris_X = iris.data #luu cac diem du lieu
iris_y = iris.target#luu cac lable 

print('Labels:', np.unique(iris_y))#tra ve cac phan tu duy nhat trong mot mang co sap xep
# tach phan mau du lieu thanh tap huan luyen va tap kiem thu
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130) 
print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])
 
#khi ket qua voi 1NN
model = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

#khi ket qua voi 7NN 
model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
print("Accuracy of 7NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

#danh gia ket qua 7NN dua tren trong so diem lan can distance 
model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance')
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
print("Accuracy of 7NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))

#danh gia ket qua 7NN dua tren trong so diem lan can Myweight
def myweight(distances): 
    sigma2 = .4 # we can change this number
    return np.exp(-distances**2/sigma2)

model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = myweight) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
print("Accuracy of 7NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))
