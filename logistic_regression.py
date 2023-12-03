#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:52:31 2023

@author: pulsaragunawardhana
"""

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df = pd.read_csv('customer_churn.csv')

print(df.head())
print(df.info())

#No null values for to be dropped. 

df.drop(['Names','Onboard_date','Location','Company'],axis=1,inplace=True)

#Dropping uneccessary attributes 

print(df.head())
print(df.info())

Y = df['Churn'].values 

X = df.drop(['Churn'],axis=1)

print(Y)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.4, random_state = 20)

model = LogisticRegression()

model.fit(X_train,y_train)

prediction_test = model.predict(X_test)

print('Accuracy', metrics.accuracy_score(y_test,prediction_test))


#Checkingß

