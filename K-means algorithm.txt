'''
Title: Implemention of the K-means clustering algorithm on the iris data set
Author: Dinuri Vishara
Date: 01/02/2023
'''

# read the iris data file into a Pandas DataFrame
import numpy as np
import pandas as pd
df = pd.read_csv("iris.csv")

# filter sepal length and sepal width data
sepaldata = df[df.columns[0:2]]

# preprocessing data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler().fit(sepaldata)
stdsepaldata = scaler.transform(sepaldata)

# fit sepal data
from sklearn.cluster import KMeans
kmeans_sepal= KMeans(n_clusters=3).fit(sepaldata)

# print labels
labels = kmeans_sepal.labels_

# centroids
centroids = kmeans_sepal.cluster_centers_
print("centroids:\n",centroids)

# predict the species of the two plants
testsepaldata = np.array([[4.6,3.0],[6.2,3.0]])
predict = kmeans_sepal.predict(testsepaldata)
print("species of the two plants:",predict)

# scatter plot
import matplotlib.pyplot as plt
fig, axs = plt.subplots()
axs.scatter(df['sepal.length'], df['sepal.width'], c= kmeans_sepal.labels_.astype(float), s=50, alpha=0.5)
axs.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)
axs.scatter(testsepaldata[:,0],testsepaldata[:,1], c='green', s=100)
# label data points
for i, txt in enumerate(labels):
    axs.annotate(txt,(sepaldata["sepal.length"][i],sepaldata["sepal.width"][i]))
# Show the original species labels
for j,lbl in enumerate(df["v_short"]):
    axs.annotate(lbl, xy=(sepaldata["sepal.length"][j],sepaldata["sepal.width"][j]), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",connectionstyle="angle3"))
plt.show()


# implement the K-means algorithm on petal length and petal width data
# filter petal length and petal width data
petaldata= df[df.columns[2:4]]

# preprocessing data
scaler= StandardScaler().fit(petaldata)
stdpetaldata = scaler.transform(petaldata)

# fit petal data
kmeans_petal = KMeans(n_clusters=3).fit(petaldata)

# print labels
labels_petal = kmeans_petal.labels_

# centroids
centroids_petal = kmeans_petal.cluster_centers_
print("centroids:\n", centroids_petal)

# predict the species of the two plants
testpetaldata = np.array([[1.5, 0.2],[4.1,1.2]])
predict_petal = kmeans_petal.predict(testpetaldata)
print("species of the two plants:",predict_petal)

# scatter plot
fig2, ax = plt.subplots()
ax.scatter(df['petal.length'], df['petal.width'], c= kmeans_petal.labels_.astype(float), s=50, alpha=0.5)
ax.scatter(centroids_petal[:, 0], centroids_petal[:, 1], c='red', s=100)
ax.scatter(testpetaldata[:,0],testpetaldata[:,1], c='green', s=100)
# label data points
for i, txt in enumerate(labels_petal):
    plt.annotate(txt,(petaldata["petal.length"][i],petaldata["petal.width"][i]))
# Show the original species labels
for j,lbl in enumerate(df["v_short"]):
    ax.annotate(lbl, xy=(petaldata["petal.length"][j],petaldata["petal.width"][j]), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3"))
plt.show()









