import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
data = pd.read_csv('diabetes.csv')
data.groupby('Outcome').mean()
X = data.drop(columns='Outcome',axis=1)
Y = data['Outcome']

#standard scalar
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#splitting dataset into training and testing
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)

#Model prediction
model = SVC(kernel='linear')
model.fit(xtrain,ytrain)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(model, open(filename, 'wb'))

#model evaluation --> accuracy score on training data
xtrain_pred = model.predict(xtrain)
print("Accuary score of training data:",accuracy_score(xtrain_pred,ytrain))

#model evaluation -->accuracy score on testing data
xtest_pred = model.predict(xtest)
print("Accuracy score on testing data:",accuracy_score(xtest_pred,ytest))