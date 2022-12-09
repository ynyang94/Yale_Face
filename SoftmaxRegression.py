#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:34:18 2021

@author: zhangyanchun (Yufeng Yang 117010351, zhangyanchun is my laptop's username)
"""

import numpy as np



class SoftmaxRegression(object):
    #initialize the parameter
    #eta: step size, epoches: # of iterations
    def __init__(self, eta=0.01, epochs=50,
                 l2=0.0,
                 #minibatches=1: gradient descent method for optimization
                 minibatches=1,
                 n_classes=None,
                 random_seed=None):

        self.eta = eta
        self.epochs = epochs
        #regularization parameter
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                #get the # of classes
                self.n_classes = np.max(y)+1
                #get the # of features
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                #parameter W
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_seed=self.random_seed)
            self.cost_ = []
            #change y into an nxn matrix like [[1,0,0,0,0,0...,0],[0,1,0,0,0,...,0]]
        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                    n_batches=self.minibatches,
                    data_ary=y,
                    shuffle=True):
                # w: n_feat x n_classes
                # b: n_classes
                # net_input, softmax and diff -> n_samples x n_classes:
                #compute Z=XW+b
                net = self._net_input(X[idx], self.w_, self.b_)
                softm = self._softmax(net)
                diff = softm - y_enc[idx]
                #compute gradient (Recall the matrix form of gradient in data processing)
                mse = np.mean(diff, axis=0)
                grad = np.dot(X[idx].T, diff)
                # update parameter towards optimization direction
                self.w_ -= (self.eta * grad +
                            self.eta * self.l2 * self.w_)
                self.b_ -= (self.eta * np.sum(diff, axis=0))

            # compute cost of the whole epoch
            net = self._net_input(X, self.w_, self.b_)
            softm = self._softmax(net)
            cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)
        return self

    def fit(self, X, y, init_params=True):
        #Learn model from training data.
        #X is the input matrix, y is label
        #init_params :initile the parameter for fitting

        
        if self.random_seed is not None:
        #set the initial parameter
            np.random.seed(self.random_seed)
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self
    
    def _predict(self, X):
        #return the probability
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
 
    def predict(self, X):
       #X here should be test data
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def predict_proba(self, X):
        
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    def _softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return - np.sum(np.log(output) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        L2_term = self.l2 * np.sum(self.w_ ** 2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        #return the label indx
        return z.argmax(axis=1)
    
    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, random_seed=None):
        #Initialize weight coefficients.
        if random_seed:
            np.random.seed(random_seed)
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        return b.astype(dtype), w.astype(dtype)
    
    def _one_hot(self, y, n_labels, dtype):
        #transform the y data into the matrix form
        #i.e y=[1,2,3,4,5]
            #transform to y= [[1,0,0,0,0],
                     #[0,1,0,0,0],
                     #[0,0,1,0,0],
                     #[0,0,0,1,0],
                     #[0,0,0,0,1]]
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            #change the corresponding position into 1
            mat[i, val] = 1
        return mat.astype(dtype)    
    
    def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
            indices = np.arange(data_ary.shape[0])

            if shuffle:
                indices = np.random.permutation(indices)

            #return the index prepared for gradient descent optimization
            minis = (indices,)

            for idx_batch in minis:
                yield idx_batch
    
    def _shuffle_arrays(self, arrays):
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]
