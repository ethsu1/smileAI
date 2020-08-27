from convolution import *
from max_pool import *
from layers import *
from relu import *
from fc import *
from loss_func import *
from flat import *
import pickle

class ConvNet:
	def __init__(self, model=[]):
		'''
		params: model - list - list of neural network layers
		'''
		self.model = model

	def forward(self, data):
		output = data
		for layer in self.model:
			output = layer.forward(output)
		return output

	def backward(self, logsoftmax, onehot, lr):
		'''
		compute backpropagation
		params:
		logsoftmax - numpy array - output of logsoftmax
		onehot - numpy array - onehot encoded ground truth labels
		lr - float - learning rate
		'''
		layer = layers()
		gradient = layer.gradient_logsoftmax(logsoftmax,onehot)
		for l in self.model[::-1]:
			try:
				gradient = l.backprop(gradient)
			except:
				gradient, _, _ = l.backprop(lr, gradient)
		return gradient
	def save_model(self, filename):
		'''
		save the model
		params:
		filename - str - filename of file saving the model
		weightname - str - filename of file saving the model's weights
		'''
		with open(filename, 'wb') as outputs:
			pickle.dump(vars(self), outputs, pickle.HIGHEST_PROTOCOL)

	def load_model(self, filename):
		with open(filename, 'rb') as inputs:
			temp = pickle.load(inputs)
			self.model = temp['model']



