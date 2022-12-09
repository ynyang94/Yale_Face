#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:49:46 2021

@author: zhangyanchun (Yufeng Yang 117010351, zhangyanchun is my laptop's user name)
"""

import numpy as np 
import csv
#define the function to read the data
def FetchData(fllename):
	with open(fllename) as f:
		train_array = np.loadtxt(f, str, skiprows = 1, delimiter = ',')

		# data preprocess
		train_array[np.where(train_array == '?')] = '0'
		class_array = train_array[:, 64]
		train_array = train_array[:, :64]

		# change data type
		train_array = train_array.astype('float')
		class_array = class_array.astype('int')

		# normalize
		train_array_norm = train_array

		return train_array_norm, class_array		
#read train and test data respectively
train3,train4=FetchData('/Users/zhangyanchun/Desktop/FTE4560/code/Bankruptcy/training.csv')
test3,test4=FetchData('/Users/zhangyanchun/Desktop/FTE4560/code/Bankruptcy/testing.csv')
#matrix scaling
X1=[]
for i in range(0,64):
    train3[:,i]=(train3[:,i]-train3[:,i].mean())/train3[:,i].std()
    X1.append(train3[:i])
X1=np.array(train3)
X_test_1=[]
for i in range(0,64):
    test3[:,i]=(test3[:,i]-test3[:,i].mean())/test3[:,i].std()
    X_test_1.append(test3[:,i])
X_test_1=np.array(X_test_1)
matrix2=[]
#repeat the experiment for 10 times
for n in range(0,10):
    result2=SoftmaxRegression(eta=0.01,epochs=10,l2=0, minibatches=1, random_seed=0)
    result2.fit(train3,train4)
    y_test_1=result2.predict(test3)
    acc2=1-abs(sum(test4-y_test_1))/151
    matrix2.append(y_test_1)
    n=n+1
#calculate the mean and variance
matrix2=np.array(matrix2)
matrix2[:,0:100].mean()
matrix2[:,0:100].var()
matrix2[:,100:150].mean()
matrix2[:,100:150].var()
    
    
#print(y_test_1.mean())
#print(np.square(y_test_1.std()))
