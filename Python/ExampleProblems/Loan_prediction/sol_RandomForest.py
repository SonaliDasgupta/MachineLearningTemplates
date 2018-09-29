# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:19:15 2018

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


def fillMissingValues(train_data):
    train_data["Gender"]=train_data["Gender"].fillna("Male")
    if(train_data["Married"].value_counts()["Yes"]> train_data["Married"].value_counts()["No"]):
        train_data["Married"]=train_data["Married"].fillna("Yes")
    else:
        train_data["Married"]=train_data["Married"].fillna("No")
    
    train_data.loc[train_data["Dependents"]=='3+','Dependents']='3'
    train_data["Dependents"].fillna(train_data["Dependents"].mode()[0], inplace= True)

    if(train_data['Self_Employed'].value_counts()["Yes"]> train_data["Self_Employed"].value_counts()["No"]):
        train_data['self_Employed'].fillna('Yes', inplace=True)
    else:
        train_data['Self_Employed'].fillna('No', inplace= True)
    
    train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(), inplace=True)
    train_data["Loan_Amount_Term"].fillna(train_data["Loan_Amount_Term"].mean(), inplace=True)
    train_data["Credit_History"].fillna(train_data["Credit_History"].mode()[0], inplace=True)
    
    

def encodeCatVars(X_train, X_test):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    le_train= LabelEncoder()
    X_train[:,0]=le_train.fit_transform(X_train[:,0])
    X_train[:,1]=le_train.fit_transform(X_train[:,1])
    X_train[:,3]=le_train.fit_transform(X_train[:,3])
    X_train[:,4]=le_train.fit_transform(X_train[:,4])
    X_train[:,-1]=le_train.fit_transform(X_train[:,-1])
    oh_train= OneHotEncoder(categorical_features=[0,1,3,4,-1])
    X_train=oh_train.fit_transform(X_train).toarray()

    le_test= LabelEncoder()
    X_test[:,0]=le_test.fit_transform(X_test[:,0])
    X_test[:,1]=le_test.fit_transform(X_test[:,1])
    X_test[:,3]=le_test.fit_transform(X_test[:,3])
    X_test[:,4]=le_test.fit_transform(X_test[:,4])
    X_test[:,-1]=le_test.fit_transform(X_test[:,-1])
        
    oh_test= OneHotEncoder(categorical_features=[0,1,3,4,-1])
    X_test=oh_test.fit_transform(X_test).toarray()
    
    return X_train, X_test
    


def scale(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train, X_test
    

train_data= pd.read_csv("D:/Analytics_Vidhya_loan_prediction/train.csv")
y_train = train_data.iloc[:,-1].values
test_data= pd.read_csv("D:/Analytics_Vidhya_loan_prediction/test.csv")
train_size= len(train_data)
test_size= len(test_data)
dataset= train_data.iloc[:,:-1].append(test_data)

fillMissingValues(dataset)

X_train= dataset.iloc[:614, 1:].values
X_test=dataset.iloc[614:,1:].values

X_train, X_test= encodeCatVars(X_train, X_test)

X_train, X_test= scale(X_train , X_test)





#Logistic Regression
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=2000) 
classifier.fit(X_train, y_train)

#predict test results
y_pred= classifier.predict(X_test)

predictions= pd.DataFrame({'Loan_ID': test_data.iloc[:,0].values , 'Loan_Status':y_pred})
predictions.to_csv("D:/Analytics_Vidhya_loan_prediction/results.csv", index=False)

#visualizing

"""def plot(X, y):
    from matplotlib.colors import ListedColormap
    X_set, y_set= X, y
    y_set[y_set=='Y']=1
    y_set[y_set=='N']=0
    classifier1= SVC(kernel='rbf')
    classifier1.fit(X_set, y_set)
    X1, X2= np.meshgrid(np.arange(start = X_set[:,9].min()-1, stop= X_set[:,9].max()+1,step=0.01),
                    np.arange(start = X_set[:,11].min()-1, stop= X_set[:,11].max()+1,step=0.01)
                   )
    plt.contourf(X1, X2, classifier1.predict(X_set), alpha=0.75, cmap= ListedColormap('red','green'))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,9], X_set[y_set==j,11], c= ListedColormap(('red','green'))(i), label=j)
    plt.title('Classifier')
    plt.xlabel('Income')
    plt.ylabel('Loan amount')
    plt.legend()
    plt.show()
    
plot(X_train, y_train)"""




