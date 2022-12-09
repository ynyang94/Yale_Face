#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:53:04 2021

@author: zhangyanchun (Yufeng Yang117010351, zhangyanchun is my laptop's username)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from random import sample





for dirname, _, filenames in os.walk('/Users/zhangyanchun/Desktop/FTE4560/code/yalefaces'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

image=plt.imread('/Users/zhangyanchun/Desktop/FTE4560/code/yalefaces/1/s6.bmp')
plt.imshow(image)
plt.show()
image.shape

dataset=[]#create a tuple to count the image data and label
labels=[]
for root,dir,files in os.walk("/Users/zhangyanchun/Desktop/FTE4560/code/yalefaces"): 

    for file in files:#read files in data
        if file!="Thumbs.db" :
            #spilt the index as label
            label=(root.split("/")[-1])
            image_path=os.path.join(root,file) 
            #vectorize the image data
            image=plt.imread(image_path) 
            #rescale the image data into 100x100
            image=image.reshape(100*100)
            #rescale the pixel value
            image=image/255
            dataset.append(image)
            #make the label come into integer
            labels.append(int(label))
            
dataset=np.array(dataset)
labels=np.array(labels)


##spilt data into test and train data randomly
train1=[]
train2=[]
test1=[]
test2=[]
#divide each subject 
for i in range(0,15,1):
    subdataset=dataset[(i*11):(i*11+11),]
    
    sublabel=labels[(i*11):(i*11+11)]
    
    l=11
    lamda=8#either 4 or 8
    #generate random index
    indices=sample(range(11),lamda)
    #get train data
    train_data=subdataset[indices]
    #get test data
    test_data=np.delete(subdataset, indices, axis=0)
    
    train_label=sublabel[indices]
    
    test_label=np.delete(sublabel,indices)
    
    train1.append(train_data)
    
    train2.append(train_label)
    
    test1.append(test_data)
    
    test2.append(test_label)
#reshape the image data into matrix form
train1=np.reshape(train1,(120,10000))#either 60 or 120
#reshape label data
train2=np.reshape(train2,(120, ))

test1=np.reshape(test1,(45,10000) )

test2=np.reshape(test2,45)

##scaling matrix
X=[]
for i in range(0,120):#either 60 or 120
    train1[i,:]=(train1[i,:]-train1[i,:].mean())
    X.append(train1[i,:])
X=np.array(X)
y=train2

X_test=[]
for i in range (0,45):#either 105 or 45
    test1[i,:]=(test1[i,:]-test1[i,:].mean())
    X_test.append(test1[i,:])
X_test=np.array(X_test)
#determine the step size, epoches, minibatch=1 indicates gradient descent 
result1=SoftmaxRegression(eta=0.01,epochs=10,l2=0, minibatches=1, random_seed=0)
#get parameters
result1.fit(X,y)
y_test=result1.predict(X_test)
#define test accuracy
def accuracy(y_train, y_test):
    n=0
    for i in range(0,45):#either 105 or 45
        if (y_train[i]-y_test[i]==0)==True:
            n=n+1
        else:
            n=n
    return n/45 #either 105 or 45
accuracy(test2, y_test)
#repeat the experiment for 10 times
y_exp10=y_test
y_exp=[y_exp1,y_exp2,y_exp3,y_exp4,y_exp5,y_exp6,y_exp7,y_exp8,y_exp9,y_exp10]
y_exp=np.array(y_exp)

y_exp_mean=[]
y_exp_var=[]
#compute the mean and variance
for i in range(1,16):
    sub_mean=y_exp[:,(i-1)*3:i*3].mean()
    sub_var=y_exp[:,(i-1)*3:i*3].var()
    y_exp_mean.append(sub_mean)
    y_exp_var.append(sub_var)

y_exp_mean=np.array(y_exp_mean)
y_exp_var=np.array(y_exp_var)
    




    
    

        
    
        
        
    
