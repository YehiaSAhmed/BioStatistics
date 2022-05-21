import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

diabetes_dataset =pd.read_csv('diabetes.csv')
pd.read_csv('diabetes.csv')
diabetes_dataset.head()

#Checking for misssing values
diabetes_dataset.isnull().sum()

diabetes_dataset.info()

#remove patient_number is data in it is redundant in predection
diabetes_dataset.drop(["patient_number"], axis = 1, inplace = True)
diabetes_dataset.head()

#chol_hdl_ratio, bmi, waist_hip_ratio are showing object so need to change.
diabetes_dataset['chol_hdl_ratio'] = pd.Series(diabetes_dataset['chol_hdl_ratio']).str.replace(',','.')
diabetes_dataset['bmi'] = pd.Series(diabetes_dataset['bmi']).str.replace(',','.')
diabetes_dataset['waist_hip_ratio'] = pd.Series(diabetes_dataset['waist_hip_ratio']).str.replace(',','.')

diabetes_dataset['chol_hdl_ratio'] = pd.to_numeric(diabetes_dataset['chol_hdl_ratio'])
diabetes_dataset['bmi'] = pd.to_numeric(diabetes_dataset['bmi'])
diabetes_dataset['waist_hip_ratio'] = pd.to_numeric(diabetes_dataset['waist_hip_ratio'])
diabetes_dataset.info()

diabetes_dataset['gender'] = pd.Series(diabetes_dataset['gender']).str.replace('female','0')
diabetes_dataset['gender'] = pd.Series(diabetes_dataset['gender']).str.replace('male','1')
diabetes_dataset['diabetes'] = pd.Series(diabetes_dataset['diabetes']).str.replace('No diabetes','0')
diabetes_dataset['diabetes'] = pd.Series(diabetes_dataset['diabetes']).str.replace('Diabetes','1')
diabetes_dataset['gender'] = pd.to_numeric(diabetes_dataset['gender'])
diabetes_dataset['diabetes'] = pd.to_numeric(diabetes_dataset['diabetes'])

diabetes_dataset.info()
diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['diabetes'].value_counts()
diabetes_dataset.groupby('diabetes').mean()
X = diabetes_dataset.drop(columns = 'diabetes', axis=1)
Y = diabetes_dataset['diabetes']

scaler = StandardScaler()
scaler.fit(X)
standardize_data = scaler.transform(X)
print(standardize_data)
X = standardize_data
Y = diabetes_dataset['diabetes']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=2)

classifier =DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)




def predection(x):
  #changing data to array
  input_data_as_numpy_array  = np.asarray(x)
  #reshaping array
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  #standarize input data
  std_data = scaler.transform(input_data_reshaped)
  reslut = classifier.predict(std_data)
  return reslut
  