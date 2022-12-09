# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:26:22 2021

@author: Golden Fish
"""

import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd

def rename(path):
	for i in os.listdir(path):
		if i.endswith('.jpg'):
			break
		elif i.endswith('.txt'):
			os.remove(path + '/' + i)
		else:
			findnum = i.index('t')
			index = i[findnum+1:findnum+3]
			separate = i.index('.')
			type = i[separate+1:]
			new_name = index + '_' + type +'.jpg'
			os.rename(path + '/' + i, path + '/' + new_name)

def readImages(path_name):
    subject = dict()
    for item in os.listdir(path_name):

    	if item.endswith('.jpg') and item != 'or_DS_Store.jpg':
    		img = Image.open(path_name + '/' + item).convert('RGB')
    		img.save(path_name + '/' + item)
    		subject_name = (item.split('.')[0]).split('_')[0]

    		img = cv2.imread(path_name + '/' + item, cv2.IMREAD_GRAYSCALE)
    		x, y = img.shape[0:2]

            # rescale the images to 64*48
    		img_resize = cv2.resize(img, (int(y / 5), int(x / 5)))

            # rescale the image pixel value
    		img_array = np.array(list(map(lambda x: x / 255, img_resize.flatten())))
    		if subject_name in subject:
    			subject[subject_name] = np.vstack([subject[subject_name], img_array])
    		else:
    			subject[subject_name] = img_array

    # vectorize every image into a column vector
    for key in subject.keys():
        subject[key] = subject[key].T

    return subject
            
def SeparateData(subject, train_num):
	train_subject = dict()
	test_subject = dict()

	for key in subject.keys():
		train_subject[key] = subject[key][:,:train_num]
		test_subject[key] = subject[key][:,train_num:]

	return train_subject, test_subject

rename('C:/Users/Golden Fish/Desktop/yaleface')
subject = readImages('C:/Users/Golden Fish/Desktop/yaleface')

train_1, test_1 = SeparateData(subject, 4)
train_2, test_2 = SeparateData(subject, 8)

X_train1 = (train_1['01']).T
for i in ['02','03','04','05','06','07','08','09','10','11','12','13','14','15']:
    X_train1 = np.r_[X_train1, (train_1[i]).T]
Y_train1 = []
repeat = 4
#['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    while repeat > 0:
        Y_train1.append(j)
        repeat -= 1
    repeat = 4

X_train2 = (train_2['01']).T
for i in ['02','03','04','05','06','07','08','09','10','11','12','13','14','15']:
    X_train2 = np.r_[X_train2, (train_2[i]).T]
Y_train2 = []
repeat = 8
for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    while repeat > 0:
        Y_train2.append(j)
        repeat -= 1
    repeat = 8
    
X_test1 = (test_1['01']).T
for i in ['02','03','04','05','06','07','08','09','10','11','12','13','14','15']:
    X_test1 = np.r_[X_test1, (test_1[i]).T]
    
Y_test1 = []
repeat = 7
for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    while repeat > 0:
        Y_test1.append(j)
        repeat -= 1
    repeat = 7

X_test2 = (test_2['01']).T
for i in ['02','03','04','05','06','07','08','09','10','11','12','13','14','15']:
    X_test2 = np.r_[X_test2, (test_2[i]).T]
    
Y_test2 = []
repeat = 3
for j in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    while repeat > 0:
        Y_test2.append(j)
        repeat -= 1
    repeat = 3

class KNNClassifier(object):
    def __init__(self, k=3):
        self.k = k

    def Distance(self, X_test, X_train):
        squareDis = (X_test - X_train) ** 2
        Dislist = []
        for i in range(X_train.shape[0]):
            Dislist.append(np.sum(squareDis[i]))
        Dislist -= np.max(Dislist)
        DisIndex = np.argsort(Dislist)
        return Dislist, DisIndex

    def classify(self, X_test, X_train, Y_train):
        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape((1, X_test.shape[0]))
        X_train = np.array(X_train)
        if len(X_train.shape) == 1:
            X_train = X_train.reshape((1, X_train.shape[0]))
        Y_train = np.array(Y_train)
        result = []
        for j in range(X_test.shape[0]):
            Dislist, DisIndex = self.Distance(X_test[j], X_train)
            classCount = {}
            kLabels = Y_train[DisIndex[: self.k]]
            for i in range(kLabels.shape[0]):
                classCount[kLabels[i]] = classCount.get(kLabels[i], 0) + 1
            valueList = list(classCount.values())
            if valueList.count(np.max(valueList)) == 1:
                result.append(kLabels[valueList.index(np.max(valueList))])
            else:
                keyList = list(classCount.keys())
                maxValue = np.max(valueList)
                maxNumberLabels = []
                i = 0
                for value in valueList:
                    if value == maxValue:
                        maxNumberLabels.append(keyList[i])
                    i += 1
                for index in range(len(kLabels)):
                    if kLabels[index] in maxNumberLabels:
                        result.append(kLabels[index])
                        break
        return result
    
def solve_eigenvalue(data,d):

    D = data.shape[1]-1
    S_w,S_b = np.zeros((D,D)),np.zeros((D,D))    
    m = (data.iloc[:,:-1].values.T).mean(axis=1)
    labels = list(set(data.iloc[:,-1].tolist()))
    for i in labels:
        subclass = data.loc[data.iloc[:,-1]==i] 
        subclass = subclass.iloc[:,:-1].values.T
        S_w += S_within_k(subclass)
        S_b += S_between(subclass,m)
    S_w = pd.DataFrame(S_w).fillna(0).values
    S_b = pd.DataFrame(S_b).fillna(0).values
    try:
        val,vec=np.linalg.eig(np.dot(np.mat(S_w).I ,S_b))
    except:
        S_w_inverse = np.linalg.pinv(np.mat(S_w))
        val,vec=np.linalg.eig(np.dot(S_w_inverse ,S_b))
    index_vec = np.argsort(-val)
    largest_index = index_vec[:d] 
    W = vec[:,largest_index]
    return W

def S_within_k(subclass):
    x_1 = (subclass.T - subclass.mean(axis=1).T).T
    return np.matmul(x_1,x_1.T)

def S_between(subclass,m):
    n = subclass.shape[1]
    x_2 = subclass.mean(axis=1) - m
    return n*np.matmul(x_2,x_2.T)


def lda_knn(X_train,X_test,Y_train,k):
    
    d = X_train.shape[1]-1
    train_data = np.c_[X_train,Y_train]
    train_data = pd.DataFrame(train_data)
    w = solve_eigenvalue(train_data,d)

    X_train = (np.dot(w.T,X_train.T).T).A
    X_test =  (np.dot(w.T,X_test.T).T).A 

    clf = KNNClassifier(k)
    result = clf.classify(X_test, X_train, Y_train)
    precision_rate = accuracy(result,Y_test2)
    return result, precision_rate

def accuracy(result,Y_test):
    error = 0
    for i in range(len(result)):
        if result[i] != Y_test[i]:
            error += 1
    precision_rate = 1 - error / len(Y_test)
    return precision_rate

if __name__ == "__main__":
    clf = KNNClassifier(k=9)
    result = clf.classify(X_test2, X_train2, Y_train2)
    knn_rate = accuracy(result,Y_test2)
    print("yalefaces KNN: ", knn_rate)
    result, lda_rate = lda_knn(X_train2,X_test2,Y_train2,k=9)
    print("yalefaces LDA+KNN: ", lda_rate)