
# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
warnings.filterwarnings('ignore')
print('-'*50)
 

#Importing DataSet
training_set    =pd.read_csv('train.csv')
testing_set        =pd.read_csv('test.csv')
final_output    =pd.read_csv('gender_submission.csv')

#dropping few columns 
training_set = training_set.drop(["Cabin","Ticket","Embarked"], axis=1)
testing_set  = testing_set.drop(["Cabin","Ticket","Embarked"], axis=1)

#encoding the Sex columns into 0 and 1
training_set['Sex'].update(training_set['Sex'].map({'male': 1, 'female': 0}))
testing_set['Sex'].update(testing_set['Sex'].map({'male': 1, 'female': 0}))

#combining sibs and parch to family
training_set['Family'] = training_set['SibSp']+training_set['Parch']
testing_set['Family']  = testing_set['SibSp']+testing_set['Parch']

#analyzing the Name column for traning data
title_count=training_set["Name"].groupby([training_set["Name"].str.split().str[1]]).nunique()

#function to replace the name column to nuerical value using title
def name_count(data):
    value=0
    if(data.split()[1] == "Mr."):
        value=5
    elif(data.split()[1] == "Mrs."):
        value=4
    elif(data.split()[1] == "Miss."):
        value=3
    elif(data.split()[1] =="Master."):
        value=2
    else:
        value=1
    return value
training_set["Name"].update(training_set["Name"].apply(name_count))
testing_set["Name"].update(testing_set["Name"].apply(name_count))

#selecting the age without nan value for predicting the nan values
age_x_y=training_set[["Name","Sex","Age","Family","Pclass"]].sort_values(by=['Age'],ascending=True).head(600)

#splitting into x and y label
age_x=age_x_y[["Name" ,"Sex","Family","Pclass"]]
age_y=age_x_y["Age"]

#splitting the dataset and applying regression
age_x_train,age_x_test,age_y_train,age_y_test=train_test_split(age_x,age_y,test_size=0.25, random_state=0)
liner_r=LinearRegression()
liner_r.fit(age_x_train,age_y_train)
pred_age=liner_r.predict(age_x_test)

#accuracy for the age column prediction
print("Accuracy for Age prediction:",metrics.r2_score(age_y_test, pred_age))

#funtion to return age based on prediction
def age_pred(data):
    name,Sex,Family,Pclass,Age = data[0],data[1],data[2],data[3],data[4]
    if(pd.isnull(Age)):
        value_array=np.array([name,Sex,Family,Pclass])
        result=liner_r.predict(value_array.reshape(1,-1))
        return int(result)
training_set["Age"].update(training_set[["Name" ,"Sex","Family","Pclass","Age"]].apply(age_pred,axis=1))
testing_set["Age"].update(testing_set[["Name" ,"Sex","Family","Pclass","Age"]].apply(age_pred,axis=1))

#heatmap for visualization
sns.heatmap(training_set.corr(),annot=True)
     
#Splitting the dataset into X and Y for Survival Prediction
X   = training_set[["Pclass","Name","Sex","Age","Fare","Family"]]
Y   = training_set["Survived"]

X_t = testing_set[["Pclass","Name","Sex","Age","Fare","Family"]]
X_t = X_t.fillna(X.mean())


#Splitting dataset to test and train for survival prediction
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.25, random_state=0)

#Fitting Random Forest Classification
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, Y_train)
Y_pred_RFC  = rfc.predict(X_test)

#checking model performance
cm=confusion_matrix(Y_test,Y_pred_RFC)
ac=accuracy_score(Y_test,Y_pred_RFC)
print("The Confusion Matrix is:\n",cm)
print("The Accuracy is :",round(ac,2)*100,'%')

#predicting the result for testing set
Pred_test = rfc.predict(X_t)

#Conveting the output to Survided_data.csv file
final_output['Survived']=pd.DataFrame(Pred_test, columns=['Survived'])
final_output.to_csv('submission.csv', index= False)





