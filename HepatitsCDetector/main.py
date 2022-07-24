import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("HepatitisCdata.csv", index_col=0)

print(data.head())

data['Category'] = data['Category'].replace(['0=Blood Donor'],'0')
data['Category'] = data['Category'].replace(['1=Hepatitis'],'1')
data['Category'] = data['Category'].replace(['2=Fibrosis'],'2')
data['Category'] = data['Category'].replace(['3=Cirrhosis'],'3')
print(data.head())
data['Category'] = pd.to_numeric(data['Category'],errors = 'coerce')
data['Sex'] = data['Sex'].replace(['m'],'1')
data['Sex'] = data['Sex'].replace(['f'],'0')
data['Sex'] = pd.to_numeric(data['Sex'],errors = 'coerce')

data = data.dropna()
print(data.isnull().sum())
print(data.dtypes)
X = data.drop("Category", axis=1)
std_data = StandardScaler()
std_data.fit(X)
X = std_data.transform(X)
Y = data['Category']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print("the accuracy percentage  with Svm is ", accuracy*100 , "%")

X_train_2 , X_test_2 , Y_train_2 , Y_test_2 = train_test_split(X,Y,test_size=0.2, random_state=3)

model_2 = KNeighborsClassifier(n_neighbors=5,p=2)
model_2.fit(X_train_2,Y_train_2)
Y_pred_2 = model_2.predict(X_test_2)
accuracy_2 = accuracy_score(Y_test_2,Y_pred_2)

print("the accuracy percentage  with KNN is ", accuracy_2*100 , "%")


X_train_3 , X_test_3 , Y_train_3, Y_test_3 = train_test_split(X,Y,test_size=0.15, random_state=4)
model_3 = LogisticRegression()
model_3.fit(X_train_3,Y_train_3)

Y_pred_3 = model.predict(X_test_3)

accuracy_3 = accuracy_score(Y_test_3,Y_pred_3)

print("the accuracy perentage with Logistic regression is : ", accuracy_3*100 ,"%")