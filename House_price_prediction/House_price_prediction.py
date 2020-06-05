# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:28:19 2020

@author: subham

"""
"Kaggle:House Price Prediction"

# Importing required libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

 
"Importing DataSet"
training_set    =pd.read_csv('train.csv')
test_set        =pd.read_csv('test.csv')
final_output    =pd.read_csv('sample_submission.csv')


print('Training_Data:', training_set.shape)
print('Test_Data:',     test_set.shape)

"removing columns with high missing value"
total_data=training_set.isnull().sum().sort_values(ascending=False)
missing_percentage=(training_set.isnull().sum()/training_set.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total_data,missing_percentage],axis=1,keys=['Total', 'Percent'])
training_set=training_set.drop(missing_data[missing_data['Total']>200].index,1)

correlation=training_set.corr()
'''f,ax=plt.subplots(figsize=(20,9))
sns.heatmap(correlation,vmax=0.8, annot=True)'''

"for training data"

categorical_data_column   =training_set.select_dtypes(include=['object']).columns
numerical_data_column     =training_set.select_dtypes(exclude=['object']).columns
categorical_data          =training_set[categorical_data_column]
numerical_data            =training_set[numerical_data_column]
numerical_data            =numerical_data.fillna(numerical_data.median())
categorical_data          =pd.get_dummies(categorical_data)

test_set_featured=numerical_data['SalePrice']
numerical_data.drop(columns='SalePrice',inplace=True)
training_set_featured=pd.concat([numerical_data,categorical_data], axis=1)
training_set_featured.drop(columns='Id',inplace=True)



"for test data"

categorical_data_column_test   =test_set.select_dtypes(include=['object']).columns
numerical_data_column_test     =test_set.select_dtypes(exclude=['object']).columns
categorical_data_test          =test_set[categorical_data_column_test]
numerical_data_test            =test_set[numerical_data_column_test]
numerical_data_test            =numerical_data.fillna(numerical_data_test.median())
categorical_data_test          =pd.get_dummies(categorical_data_test)

training_set_featured_test=pd.concat([numerical_data_test,categorical_data_test], axis=1)
training_set_featured_test=training_set_featured_test.iloc[:-1,:]
training_set_featured_test.drop(columns='Id',inplace=True)


X_train,X_test,Y_train,Y_test=train_test_split(training_set_featured,test_set_featured,test_size=0.3, random_state=0)


from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas = [ 0.6,0.7,0.75,0.76,0.8, 1, 3, ])
ridge.fit(X_train,Y_train)
alpha = ridge.alpha_
print('best alpha',alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 7)
ridge.fit(X_train, Y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)
y_train_rdg = ridge.predict(training_set_featured_test)


"Conveting the output to Survided_data.csv file "
final_output['SalePrice']=pd.DataFrame(y_train_rdg, columns=['SalePrice'])
final_output.to_csv('submission.csv', index= False)


print('Program Completed')