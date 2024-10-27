'''
Title: Implemention of K-nearest neighbors (KNN) algorithm.
Author: Dinuri Vishara
Date: 01/02/2023
'''
# read the bundled iris data set in the scikit-learn package
from sklearn import datasets
iris= datasets.load_iris()

# load the iris data
irdata = iris.data

# extract the labels
irlabel = iris.target

# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(irdata)
stdirdata = scaler.transform(irdata)

# standardize plant 1 data
import numpy as np
plant1 = np.array([[4.6, 3.0, 1.5, 0.2]])
scaler = StandardScaler().fit(plant1)
stdplant1 = scaler.transform(plant1)

# create a 2-d array for the two data records
testdata = np.array([[4.6, 3.0, 1.5, 0.2],[6.2, 3.0,4.1,1.2]],dtype=np.float64)
# standardize the 2D array
scaler = StandardScaler().fit(testdata)
stdtestdata = scaler.transform(testdata)
print("standardized 2D array:\n", stdtestdata)

# extract the standardized sepal data
sepaldata = stdirdata[:,[0,1]]
# print(sepaldata)

# train the KNN model to find the two nearest neighbors of the test data record
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=2).fit(sepaldata)
print(nn)

# find the two-nearest neighbors
distances, indices = nn.kneighbors(stdtestdata[:,[0,1]])

# print indices of two-nearest neighbors
print("indices of two-nearest neighbors:\n",indices)
# print data value of two-nearest neighbors
print("data value of two-nearest neighbors:\n",iris["data"][indices])
# print labels of two-nearest neighbors
label = iris["target"][indices]
print("labels of two-nearest neighbors:\n",label)
# print species of two-nearest neighbors
print("species of two-nearest neighbors:\n",iris["target_names"][label])

# train the model for 5-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
nn_5 = KNeighborsClassifier(n_neighbors=5).fit(sepaldata,irlabel)
dist, ind = nn_5.kneighbors(stdtestdata[:,[0,1]])

# predict the probability of each plant categorizing into each species
print("Probability of each plant categorizing into each species:\n",nn_5.predict_proba(testdata[:,[0,1]]))

# print labels of five-nearest neighbors
labels = iris["target"][ind]
print("labels of five-nearest neighbors:\n",labels)

# print species of five-nearest neighbors
print("species of five-nearest neighbors:\n",iris["target_names"][labels])



## train the KNN model using petal measurements
# extract the standardized petal data
petaldata = stdirdata[:,[2,3]]
# print(petaldata)

# train the KNN model to find the two nearest neighbors of the test data record
nn_petal = NearestNeighbors(n_neighbors=2).fit(petaldata)
# print(nn_petal)

# find the two-nearest neighbors
distances_p, indices_p = nn_petal.kneighbors(stdtestdata[:,[2,3]])

# print indices of two-nearest neighbors
print("indices of two-nearest neighbors:\n",indices_p)

# print data value of two-nearest neighbors
print("data value of two-nearest neighbors:\n",iris["data"][indices_p])

# print labels of two-nearest neighbors
label_p = iris["target"][indices_p]
print("labels of two-nearest neighbors:\n",label_p)
# print species of two-nearest neighbors
print("species of two-nearest neighbors:\n",iris["target_names"][label_p])

# train the model for 5-nearest neighbors
nn_petal5 = KNeighborsClassifier(n_neighbors=5).fit(petaldata,irlabel)
distances_p5, indices_p5 = nn_petal5.kneighbors(stdtestdata[:,[0,1]])
print(nn_petal5)

# predict the probability of each plant categorizing into each species
print("Probability of each plant categorizing into each species:\n",nn_petal5.predict_proba(testdata[:,[2,3]]))

# print labels of five-nearest neighbors
label_p5 = iris["target"][indices_p5]
print("labels of five-nearest neighbors:\n",label_p5)

# print species of five-nearest neighbors
print("species of five-nearest neighbors:\n",iris["target_names"][label_p5])


## train the KNN model using both the sepal and petal measurements
# train the KNN model to find the two nearest neighbors of the test data record
knn = NearestNeighbors(n_neighbors=2).fit(irdata)
print(knn)

# find the two-nearest neighbors
distances_knn, indices_knn = knn.kneighbors(stdtestdata)

# print indices of two-nearest neighbors
print("indices of two-nearest neighbors:\n",indices_knn)

# print data value of two-nearest neighbors
print("data value of two-nearest neighbors:\n",iris["data"][indices_knn])

# print labels of two-nearest neighbors
label_knn = iris["target"][indices_knn]
print("labels of two-nearest neighbors:\n",label_knn)

# print species of two-nearest neighbors
print("species of two-nearest neighbors:\n",iris["target_names"][label_knn])

# train the model for 5-nearest neighbors
knn_5 = KNeighborsClassifier(n_neighbors=5).fit(stdirdata,irlabel)
print(knn_5)
distknn, indknn = knn_5.kneighbors(stdtestdata)

# predict the probability of each plant categorizing into each species
print("Probability of each plant categorizing into each species:\n",knn_5.predict_proba(testdata))

# print labels of five-nearest neighbors
labels_knn = iris["target"][indknn]
print("labels of five-nearest neighbors:\n:",labels_knn)

# print species of five-nearest neighbors
print("species of five-nearest neighbors:\n",iris["target_names"][labels_knn])









