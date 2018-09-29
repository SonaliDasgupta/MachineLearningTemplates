# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:34:17 2018

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

data= pd.read_csv("D:/Udemy_ML/Logistic_Regression/Logistic_Regression/Social_Network_Ads.csv")
X = data.iloc[:,[2,3]].values
y= data.iloc[:,-1].values

#splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/4, random_state=0)

#Encoding
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_train= LabelEncoder()
X_train[:,0]=le_train.fit_transform(X_train[:,0])
oh_train= OneHotEncoder(categorical_features=[0])
X_train=oh_train.fit_transform(X_train).toarray()

le_test= LabelEncoder()
X_test[:,0]=le_test.fit_transform(X_test[:,0])
oh_test= OneHotEncoder(categorical_features=[0])
X_test=oh_test.fit_transform(X_test).toarray()"""

#Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Logistic Regression
from sklearn.svm import SVC
classifier = SVC(kernel ='linear', random_state=0)
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
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap= ListedColormap('red','green'))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1], c= ListedColormap(('red','green'))(i), label=j)
    plt.title('SVM')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()
    
plot(X_train, y_train)
plot(X_test, y_test)



