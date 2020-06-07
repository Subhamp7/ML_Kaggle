# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:57:35 2020

@author: subham
"""

#importing required libraries
import numpy as np
import pandas as pd
from   keras.models import Sequential
from   keras.layers import Dense
from   notification_sound import sound
from   keras.utils import np_utils

#importing the datasets
dataset_train =pd.read_csv('train.csv')
dataset_test  =pd.read_csv('test.csv')
final_output  =pd.read_csv('submission.csv')

#printing few information about the data
print(('The size of Training data {} \nThe size of Test data     {} ').
      format(dataset_train.shape,dataset_test.shape))
print(dataset_test.info())

#data preprocessing
dataset_train_X=dataset_train.iloc[:,1:-1]
dataset_train_Y=dataset_train.iloc[:, -1:]
dataset_train_Y = np_utils.to_categorical(dataset_train_Y)

#for testing data
dataset_test=dataset_test.iloc[:,1:]

#normalization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
dataset_train_X=sc.fit_transform(dataset_train_X)

#initilizing ANN
classifier=Sequential()
classifier.add(Dense(units=28, kernel_initializer= 'uniform', activation ='selu', input_dim=54))
classifier.add(Dense(units=14, kernel_initializer= 'uniform', activation = 'softplus'))
classifier.add(Dense(units=8, kernel_initializer= 'uniform', activation='softmax'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#fitting the ANN
classifier.fit(dataset_train_X, dataset_train_Y, batch_size=32, epochs=500)

#predicting the value for the test data
Y_pred=classifier.predict(dataset_test)
Y_pred=np.argmax(Y_pred,axis=1)

#converting the output to subission excel file
final_output['Cover_Type']=pd.DataFrame(Y_pred, columns=['Cover_Type'])
final_output.to_csv('submission.csv', index= False)

#printing after completion
print('Completed')
sound(5)
