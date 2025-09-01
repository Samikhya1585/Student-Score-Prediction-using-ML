#importing the libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,f1_score,r2_score,accuracy_score,precision_score,recall_score,confusion_matrix

#load the dataset
data=pd.read_csv("student_performance_dataset.csv")



#check for any missing data
print(data.isnull().sum()) 



#drop the stu_id column as its not needed in ML
data=data.drop("Student_ID",axis=1)
print(data.info())



#labelEncoding
label_cols=["Pass_Fail","Internet_Access_at_Home","Extracurricular_Activities"]
le=LabelEncoder()

for col in label_cols:
    data[col]=le.fit_transform(data[col])
    



#oneHotEncoding 
data=pd.get_dummies(data,columns=["Parental_Education_Level","Gender"])
print(data.head())




#numerical feature scaling
Scaler=StandardScaler()
scalar_data_col=["Study_Hours_per_Week","Attendance_Rate","Past_Exam_Scores"]
data[scalar_data_col]=Scaler.fit_transform(data[scalar_data_col])




#correlation graph
plt.figure(figsize=(10,6))
heatmap_data=data[scalar_data_col +["Final_Exam_Score"]]
sns.heatmap(heatmap_data.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")



#study hours distribution
plt.figure(figsize=(10,6))
sns.histplot(data["Study_Hours_per_Week"],kde=True)
plt.title("Distribution of study Hours")




#study_hours vs pass or fail
plt.figure(figsize=(10,6))
sns.boxplot(x="Pass_Fail",y="Study_Hours_per_Week",data=data)
plt.title("Study Hours VS Pass/Fail")
plt.show()



#Linear Regression (Predicting FinalExamScore)
x=data.drop(["Final_Exam_Score","Pass_Fail"],axis=1)
y=data["Final_Exam_Score"]

LiReg=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

LiReg.fit(X_train,y_train)
y_pred=LiReg.predict(X_test)

print("Regression metrics:")
print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE: ",mean_squared_error(y_test,y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score:",r2_score(y_test,y_pred))
print("-------------------------------------------------------------------------")




#predicting pass/fail using logistic regression

x=data.drop(["Final_Exam_Score","Pass_Fail"],axis=1)
y=data["Pass_Fail"]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
LoReg=LogisticRegression()
LoReg.fit(X_train,y_train)
y_pred=LoReg.predict(X_test)

print("Classification metics with Logistic Regression:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))




#KNN 

knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)
print("KNN Accuracy:", knn_model.score(X_test,y_test))




#Decision Tree

dec_model=DecisionTreeClassifier()
dec_model.fit(X_train,y_train)
print("Decision Accuracy:", dec_model.score(X_test,y_test))
