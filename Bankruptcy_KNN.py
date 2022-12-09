# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 01:19:57 2021

@author: Golden Fish
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing


# 导入数据集 train
from pandas import read_csv
filename1='C://Users//Golden Fish//Desktop//Project 1 -FTE4560//Project 1 of FTE4560//Bankruptcy//training.csv'

train=read_csv(filename1,header=0)

# 建立X Y 变量

train_data=np.array(train)

X_train_pre = train_data[:,0:63]
Y_train = train_data[:,64]

# 缺失值处理
from sklearn.impute import SimpleImputer

df1=pd.DataFrame(X_train_pre)
nan_all=df1.isnull()
nan_col=df1.isnull().any(0)

nan_model=SimpleImputer(missing_values=np.nan,strategy='mean') #建立替换规则：将值为nan的缺失值用均值替换
X_train_nonan=nan_model.fit_transform(df1)
X_train = X_train_nonan / X_train_nonan.max(axis=0)
zscore = preprocessing.StandardScaler()
# 标准化处理
X_train = zscore.fit_transform(X_train)


# 导入数据集 test
filename2="C:/Users/Golden Fish/Desktop/Project 1 -FTE4560/Project 1 of FTE4560/Bankruptcy/testing.csv"

test=read_csv(filename2,header=0)

# 建立X Y 变量
test_data=np.array(test)

X_test_pre = test_data[:,0:63]
Y_test = test_data[:,64]

# 缺失值处理
X_test_pre[np.where(X_test_pre == '?')] = 'NaN'
df2=pd.DataFrame(X_test_pre)

nan_model2=SimpleImputer(missing_values=np.nan,strategy='mean') #建立替换规则：将值为nan的缺失值用均值替换
X_test_nonan=nan_model2.fit_transform(df2)
X_test = X_test_nonan / X_test_nonan.max(axis=0)


#Z-Score标准化
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
X_test = zscore.fit_transform(X_test)


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
    precision_rate = accuracy(result,Y_test)
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
    result = clf.classify(X_test, X_train, Y_train)
    knn_rate = accuracy(result,Y_test)
    print("Bankruptcy KNN: ", knn_rate)
    result, lda_rate = lda_knn(X_train,X_test,Y_train,k=9)
    print("Bankruptcy LDA+KNN: ", lda_rate)