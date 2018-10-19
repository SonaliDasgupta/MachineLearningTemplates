# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:38:37 2018

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:20:41 2018

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 14:40:12 2018

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("D:\\Udemy_ML\\Kernel_PCA\\Social_Network_Ads.csv")
X = data.iloc[:,1:len(data.columns)-1].values
y= data.iloc[:,-1].values

#splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
X_train[:,0] = le_1.fit_transform(X_train[:,0])
X_test[:,0] = le_1.transform(X_test[:,0])
#oh_1 = OneHotEncoder(categorical_features=[1])
#X_train = oh_1.fit_transform(X_train)
#X_test = oh_1.transform(X_test)



#Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 1:3] = sc_X.fit_transform(X_train[:, 1:3])
X_test[:,1:3] = sc_X.transform(X_test[:,1:3])

#Apply Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train, y_train)
X_test = kpca.transform(X_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)



#predict test results
y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visualizing

def plot(X, y):
    from matplotlib.colors import ListedColormap
    X_set, y_set= X, y
    X1, X2= np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop= X_set[:,0].max()+1,step=0.01),
                    np.arange(start = X_set[:,1].min()-1, stop= X_set[:,1].max()+1,step=0.01)
                   )
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap= ListedColormap(('red','green','blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1], c= ListedColormap(('red','green','blue'))(i), label=j)
    plt.title('Logistic Regression')
    plt.xlabel('Var 1')
    plt.ylabel('Var 2')
    plt.legend()
    plt.show()
    
plot(X_train, y_train)
plot(X_test, y_test)



