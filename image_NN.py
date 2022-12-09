#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:37:33 2021
@author: Deng Qingwen
"""
import os
import cv2
from PIL import Image
import numpy as np
import scipy.special

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

class neuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, lam, num_samples):

		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes

		# initialize the weight
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.4), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.4), (self.onodes, self.hnodes))

		# set the learning rate
		self.lr = learning_rate

		# set the lambda rate
		self.lam = lam

		# set the number of samples in training dataset
		self.n = num_samples
		
		# sigmoid function as the activation function
		self.activation_function = lambda x: scipy.special.expit(x)

	def train(self, inputs, targets):

		inputs = np.array(inputs, ndmin=2).T
		targets = np.array(targets, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		# self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
		# self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
		self.who = (1 - self.lr * self.lam / self.n) * self.who + self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
		self.wih = (1 - self.lr * self.lam / self.n) * self.wih + self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

		pass

	def query(self, inputs):

		inputs = np.array(inputs, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

def main(n, epochs, train_dict, test_dict):
	for e in range(epochs):
		for key in train_dict.keys():
			for i in range(train_dict[key].shape[1]):
				targets = np.zeros(output_nodes) + 0.01
				targets[int(key)-1] = 0.99
				n.train(train_dict[key][:,i], targets)

	correct = 0
	wrong = 0
	for key in test_dict.keys():
		for j in range(test_dict[key].shape[1]):
			final_outputs = n.query(test_dict[key][:,j])
			if np.argmax(final_outputs) + 1 == int(key):
				correct += 1
			else: wrong += 1
	print('number of hidden nodes:', n.hnodes)
	print('number of training image per subject:', train_dict[key].shape[1])
	print('The precision is: {:.2%}'.format(correct/(correct+wrong)))
	print('---------------------------------------------')

rename('/Users/eunice/Downloads/FTE4560/Project_1/yaleface')
subject = readImages('/Users/eunice/Downloads/FTE4560/Project_1/yaleface')

train_1, test_1 = SeparateData(subject, 4)
train_2, test_2 = SeparateData(subject, 8)

# number of input, hidden and output nodes
input_nodes = 3072
hidden_nodes_1 = 100
hidden_nodes_2 = 200
output_nodes = 15

# learning rate
learning_rate = 0.01

# regularization parameter
lam = 1

n1 = neuralNetwork(input_nodes, hidden_nodes_1, output_nodes, learning_rate, lam, 60)
n2 = neuralNetwork(input_nodes, hidden_nodes_1, output_nodes, learning_rate, lam, 120)
n3 = neuralNetwork(input_nodes, hidden_nodes_2, output_nodes, learning_rate, lam, 60)
n4 = neuralNetwork(input_nodes, hidden_nodes_2, output_nodes, learning_rate, lam, 120)

epochs = 100

main(n1, epochs, train_1, test_1)
main(n2, epochs, train_2, test_2)
main(n3, epochs, train_1, test_1)
main(n4, epochs, train_2, test_2)
