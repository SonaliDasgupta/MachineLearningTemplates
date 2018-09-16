# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:13:00 2018

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv("D:\\Udemy_ML\\Polynomial_linear_regression\\Polynomial_Regression\\Position_Salaries.csv")
X= dataset.iloc[:,1:-1].values
y= dataset.iloc[:,len(dataset.columns)-1].values

"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])"""

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[3])
labelEncoder_X= LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
X=oneHotEncoder.fit_transform(X).toarray()"""


#avoiding dummy var trap
"""X=X[:,1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)"""

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression

#non-linear regression
#create regressor



#predict - nonlinear reg
y_pred=regressor.predict(6.5)



#visualize reg
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualize reg higher resolution
X_grid= np.arange(min(X), max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()












