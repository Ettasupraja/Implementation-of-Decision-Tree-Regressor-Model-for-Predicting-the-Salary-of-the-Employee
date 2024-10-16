# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array. 8.Apply to new unknown values.
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ETTA SUPRAJA
RegisterNumber:212223220022

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

dt.predict([[5,6]])plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()  
*/
```

## Output
![image](https://github.com/user-attachments/assets/21bff41f-3358-4025-aba3-02a9e21f3efd)
![image](https://github.com/user-attachments/assets/d2f53cf7-f2c3-4e8d-b233-40842da93cdd)
![image](https://github.com/user-attachments/assets/149ad2a7-4d86-4953-9562-6de78f98bab3)
![image](https://github.com/user-attachments/assets/6801b851-eaac-4222-a8cc-d599f58f8b0b)
![image](https://github.com/user-attachments/assets/48a1f2ba-50da-4868-87f8-ec035909f968)
![image](https://github.com/user-attachments/assets/3a9da77c-d8d0-4b97-9572-259c7110985d)
![image](https://github.com/user-attachments/assets/aa4abe3b-972d-4e4d-8f67-f5aea9a7cd69)
![image](https://github.com/user-attachments/assets/1cb81ab7-73ec-485a-9ad4-3b5db1f0a082)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
