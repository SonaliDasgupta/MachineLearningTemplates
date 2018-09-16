# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:\\Udemy_ML\\Multiple_Linear_Reg\\Multiple_Linear_Regression\\50_Startups.csv")
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,len(dataset.columns)-1].values

"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[3])
labelEncoder_X= LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
X=oneHotEncoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=1/3, random_state=0)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#multiple linear regression










