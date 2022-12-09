# -*- coding=utf-8 -*-
import numpy as np 
import csv

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
		train_array_norm = train_array / train_array.max(axis=0)

		return train_array_norm, class_array