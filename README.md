# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import pandas for data manipulation and sklearn for machine learning operations.
2.  Load data from a CSV file using pandas, then preprocess it by removing unnecessary columns and handling missing values if any.
3.  Divide the preprocessed data into training and testing sets.
4.  Train a machine learning model, such as logistic regression (lr), on the training data.
5.  Calculate accuracy, generate confusion matrix, and produce a classification report to assess model performance.
6.  Utilize the trained model to make predictions on new data points, ensuring it's fitted on training data before predicting on the test set.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHARMITHA V
RegisterNumber:212223110048
import pandas as pd
data = pd.read_csv("/content/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender'] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["speacialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![image](https://github.com/sharmitha3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145974496/fa4ebf38-aa07-4829-be75-9e77a6cec8ca)

![image](https://github.com/sharmitha3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145974496/77891a54-49b0-4549-a80a-9c3cf82ed3b8)

![image](https://github.com/sharmitha3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145974496/c6faa62f-e5bf-4cb6-8705-40b84ed9a586)

![image](https://github.com/sharmitha3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145974496/af90b508-591b-4250-b3f4-6ec17003050e)


![image](https://github.com/sharmitha3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145974496/163f64eb-ec56-4a03-aaaf-55764f1b4ee9)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
