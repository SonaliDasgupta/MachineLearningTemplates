# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:48:01 2018

@author: Admin
"""

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

#using Dendogram Method to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method= 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Distances')
plt.show()

#applying Hierarchial Clustering
from sklearn.cluster import AgglomerativeClustering
hc =AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#visualoze clusters
plt.scatter(X[y_hc==0, 0], X[y_hc==0,1], s=100, c='red', label='Careful')
plt.scatter(X[y_hc==1, 0], X[y_hc==1,1], s=100, c='blue', label='Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc==2,1], s=100, c='green', label='Target')
plt.scatter(X[y_hc==3, 0], X[y_hc==3,1], s=100, c='yellow', label='Careless')
plt.scatter(X[y_hc==4, 0], X[y_hc==4,1], s=100, c='purple', label='Sensible')

plt.legend()
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
