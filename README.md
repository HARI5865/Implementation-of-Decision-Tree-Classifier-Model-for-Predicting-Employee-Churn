# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARIHARASUDHAN N
RegisterNumber: 212224040102
```
## Data head:
```
import pandas as pd
data=pd.read_csv("Employee.csv")
display(data.head())
```

## OUTPUT:
<img width="1650" height="259" alt="image" src="https://github.com/user-attachments/assets/0b6d9cc5-a7e4-4cea-8063-91e46898d493" />

## Dataset info:
```
data.info()
```
## OUTPUT:
<img width="771" height="371" alt="image" src="https://github.com/user-attachments/assets/f872628d-fa31-49dc-abfe-7638ceee11dc" />

## Null dataset:
```
display(data.isnull().sum())
```

## OUTPUT:
<img width="648" height="467" alt="image" src="https://github.com/user-attachments/assets/0338afe2-4e95-419a-8e89-42630af55384" />

## Values in left column:
```
display(data['left'].value_counts())
```
## OUTPUT:
<img width="598" height="207" alt="image" src="https://github.com/user-attachments/assets/bedc76d2-652f-4802-a338-036d3350bcdb" />

## Prediction calculating code:
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
```
## OUTPUT:
<img width="797" height="371" alt="image" src="https://github.com/user-attachments/assets/b42b7569-555b-416a-867f-6768cd3a2550" />

## Accuracy:
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
print(accuracy)
```

## OUTPUT:
<img width="661" height="133" alt="image" src="https://github.com/user-attachments/assets/6d9c64c6-0b69-436f-b223-c9bcdbda36e1" />

## Prediction:
```
print(dt.predict([[0.5,0.8,9,206,6,0,1,2]]))
```

## Output:
<img width="1730" height="78" alt="image" src="https://github.com/user-attachments/assets/58305550-f466-4cb7-b444-2f395e322d17" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
