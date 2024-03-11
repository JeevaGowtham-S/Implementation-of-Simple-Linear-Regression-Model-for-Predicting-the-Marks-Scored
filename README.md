# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
``` py
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JEEVAGOWTHAM S
RegisterNumber:  212222230053

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
### df.head():
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/ad734e9a-22c7-4cd2-bebe-71c48b23ab6a)

### df.tail():
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/b71b9c91-0634-4f26-9949-404f54156f8a)


### Array value of X:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/0993d160-b0b3-4f43-b4ad-a7348a04a489)


### Array value of Y:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/93811d4e-4cca-4c63-ab77-594d8eaa45c2)



### Values of Y prediction:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/fa6f8067-88ea-48da-8307-cde4077edcfd)



### Values of Y test:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/b472729a-8171-4a26-99bf-3e3e3e007d97)



### Training Set Graph:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/8f2e90b5-36ae-44c6-85a0-0afcab4c42c6)



### Test Set Graph:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/79a2bdaf-7b83-4990-bed6-6a641b425456)



### Values of MSE, MAE and RMSE:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118042624/f9d8122f-8cfc-43fc-a6ab-025242845515)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
