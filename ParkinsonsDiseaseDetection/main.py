import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

data = pd.read_csv("parkinsons.csv")
print(data.head())
print(data.shape)
print(data.info())

print(data.isnull().sum())

X = data.drop(columns=['name', 'status'], axis=1)
print(X.head())

Y = data['status']
print(Y.head())


X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size= 0.15, random_state=5)

scalled_data = StandardScaler()
scalled_data.fit(X_train)

X_train = scalled_data.transform(X_train)

X_test = scalled_data.transform(X_test)

print(X_train)

model = svm.SVC(kernel='linear')

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print(accuracy)

# predictive system

user_input = (119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)

user_input_as_nparray = np.asarray(user_input)

reshaped = user_input_as_nparray.reshape(1,-1)

Std_user_data = scalled_data.transform(reshaped)

user_prediction = model.predict(Std_user_data)

print(user_prediction)

if ( user_prediction[0] == 1):
    print("parkinson's disease detected ")

else:
    print("parkinson's disease not detected ")