import pandas as pd
dataset = pd.read_csv("../input/Churn_Modelling.csv")

X = dataset.iloc[:,3:-1].values
y= dataset.iloc[:,-1].values


#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_1= LabelEncoder()
X[:,1]=le_1.fit_transform(X[:,1])
le_2 = LabelEncoder()
X[:,2]=le_2.fit_transform(X[:,2])
oh= OneHotEncoder(categorical_features=[1])
X=oh.fit_transform(X).toarray()

X = X[:,1:]


#splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#fitting XGBoost to training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#test set predictions
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train,cv=10)
accuracies.mean()
accuracies.std()