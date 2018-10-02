# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:21:25 2018

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("D:/Udemy_ML/K_Means/Mall_Customers.csv")
X= dataset.iloc[:,3:5].values

#using Elbow Method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Kmeans WCSS for clusters')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

#applying KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualoze clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0,1], s=100, c='red', label='Careful')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1,1], s=100, c='blue', label='Standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2,1], s=100, c='green', label='Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3,1], s=100, c='yellow', label='Careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4,1], s=100, c='purple', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='magenta', label='Centroids')
plt.legend()
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
