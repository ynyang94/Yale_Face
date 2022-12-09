#!/usr/bin/python
"""
Created on Mon Mar 22 22:23:33 2021
@author: Deng Qingwen
"""
import os
import cv2
import numpy as np
import scipy.special

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

		train_array_scaled = ( train_array - train_array.mean(axis=0) ) / train_array.std(axis=0)

		return train_array_scaled.T, class_array

class neuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, lam, num_samples):

		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes

		# initialize the weight
		self.wih = np.random.normal(-0.05, pow(self.hnodes, -0.4), (self.hnodes, self.inodes)) * 0.4
		self.who = np.random.normal(-0.05, pow(self.onodes, -0.4), (self.onodes, self.hnodes)) * 0.4

		# set the learning rate
		self.lr = learning_rate
		
		# set the lambda rate
		self.lam = lam

		# set the number of samples in training dataset
		self.n = num_samples
		
		# sigmoid function as the activation function
		self.activation_function = lambda x: scipy.special.expit(x)
		self.inverse_activation_function = lambda x: scipy.special.logit(x)

	def train(self, inputs, targets):

		inputs = np.array(inputs, ndmin=2).T
		targets = np.array(targets, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

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

def main(n, epochs, train_array, class_array, test_array, ideal_array):
	for e in range(epochs):
		for i in range(train_array.shape[1]):
			targets = class_array[i]
			n.train(train_array[:,i], targets)

	correct = 0
	wrong = 0
	for j in range(test_array.shape[1]):
		final_outputs = n.query(test_array[:,j])
		if round(final_outputs[0][0]) == ideal_array[j]:
			correct += 1
		else: wrong += 1

	print('number of hidden nodes:', n.hnodes)
	print('The precision is: {:.2%}'.format(correct/(correct+wrong)))


# process data
train_array, class_array = FetchData('Bankruptcy/training.csv')
test_array, ideal_array = FetchData('Bankruptcy/testing.csv')

# number of input, hidden and output nodes
input_nodes = 64
hidden_nodes_1 = 16
hidden_nodes_2 = 24
output_nodes = 1

# learning rate
learning_rate = 0.01

epochs = 10

# regularization parameter
lam = 0.1

n1 = neuralNetwork(input_nodes, hidden_nodes_1, output_nodes, learning_rate, lam, 949)
main(n1, epochs, train_array, class_array, test_array, ideal_array)

n2 = neuralNetwork(input_nodes, hidden_nodes_2, output_nodes, learning_rate, lam, 949)
main(n2, epochs, train_array, class_array, test_array, ideal_array)
