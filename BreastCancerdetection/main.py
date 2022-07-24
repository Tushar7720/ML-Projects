import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


BreastCancer_data = pd.read_csv("data.csv")
print(BreastCancer_data.head())

BreastCancer_data = BreastCancer_data.drop(["id"],axis=1)

print(BreastCancer_data.head())

print(BreastCancer_data.isnull().sum())

print(BreastCancer_data.info())

BreastCancer_data["diagnosis"] = BreastCancer_data["diagnosis"].replace("M", 0)
BreastCancer_data["diagnosis"] = BreastCancer_data["diagnosis"].replace("B", 1)
BreastCancer_data["diagnosis"].astype(float)

Y = BreastCancer_data["diagnosis"]
print(Y)

X = BreastCancer_data.drop(["diagnosis"], axis = 1)
print(X)

std_data = StandardScaler()
std_data.fit(X)
X = std_data.transform(X)
print(X)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.1, random_state=2)
model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
acurracy = accuracy_score(Y_test, Y_pred)

print("Th acurracy for Support Vector Classifier is :" , acurracy*100 , "%")


X_train_2 , X_test_2 , Y_train_2 , Y_test_2 = train_test_split(X,Y,test_size=0.2, random_state=2)
model_2 = LogisticRegression()
model_2.fit(X_train_2,Y_train_2)
Y_pred_2 = model.predict(X_test_2)
acurracy_2 = accuracy_score(Y_test_2, Y_pred_2 )

print("The acurracy for Logistic regression is :" , acurracy_2*100 , "%")

X_train_3 , X_test_3 , Y_train_3 , Y_test_3 = train_test_split(X,Y,test_size=0.2, random_state=3)
model_3 = KNeighborsClassifier(n_neighbors= 10, p=2)
model_3.fit(X_train_3,Y_train_3)
Y_pred_3 = model_3.predict(X_test_3)
acurracy_3 = accuracy_score(Y_test_3,Y_pred_3)

print("The acurracy for Knn classifier is :" , acurracy_3*100 , "%")

