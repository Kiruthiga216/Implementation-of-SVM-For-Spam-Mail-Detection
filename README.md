# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset and label messages as spam or not spam.

2.Preprocess the text (remove punctuation, lowercase, and convert to numerical form using TF-IDF).

3.Split the data into training and testing sets.

4.Train an SVM classifier (e.g., LinearSVC) on the training data.

5.Predict and evaluate performance using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:B.Kiruthiga
RegisterNumber:212224040160 
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
print(data.head())

print(data.shape)

x=data['v2'].values
y=data['v1'].values
print(x.shape)

print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

print(x_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("y_pred:\n",y_pred)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print("accuracy:\n",acc)

con=confusion_matrix(y_test,y_pred)
print("confusion_matrix:\n",con)

cl=classification_report(y_test,y_pred)
print("classification:\n",cl)


```

## Output:

<img width="702" height="719" alt="image" src="https://github.com/user-attachments/assets/2c526480-211b-4943-955a-b7a93ac54250" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
