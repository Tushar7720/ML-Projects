import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler

Heart_data = pd.read_csv("Heart Attack Data Set.csv")
print(Heart_data.head())

X = Heart_data.drop(columns=["target"])
Y = Heart_data["target"]

print(X.head())
print(Y.head())

st_model = StandardScaler()
st_model.fit(X)
X = st_model.transform(X)

print(X)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)

model = svm.SVC(kernel="linear")
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print("The accuracy score of this model is :" , accuracy*100 , "%")