import numpy as np
import math
class fc:
	def __init__(self, input_dim, output_dim, bias=True):
		'''
		Fully connected layer (which is just a linear transformation)
		weights and biases are initialized from a uniform distribution U(-sqrt(k), sqrt(k)) where k = 1/input_dim
		params: 
		input_dim - int - input dimension 
		output_dim - int - output dimension
		'''
		#self.weights = -np.random.random_sample((input_dim, output_dim))
		#self.bias = -np.random.random_sample((output_dim,))
		k = 1/input_dim
		self.weights = np.random.uniform(low=-math.sqrt(k),high=math.sqrt(k), size=(input_dim, output_dim))
		self.is_bias = bias
		if(self.is_bias == False):
			self.bias = np.zeros((output_dim,))
		else:
			self.bias = np.random.uniform(low=-math.sqrt(k),high=math.sqrt(k), size=(output_dim,))
		self.prev_output = None
		self.prev_input = None
		self.prev_shape = None

	def forward(self,matrix):
		'''
		params: matrix - numpy array - img/matrix of 2D shape  or flat vector
		'''
		output = np.matmul(matrix, self.weights) + self.bias
		self.prev_output = output
		self.prev_input = matrix
		return output

	def backprop(self, lr, gradient):
		'''
		perform backpropagation to adjust the weights and biases of the fully connected layer
		returns gradient to pass to other layers
		return:
		d_loss_d_x - numpy array - gradient of loss with respect to fully connected layer inputs
		d_loss_d_weights - numpy array - gradient of loss with respect to weights
		d_loss_d_bias - numpy array - gradient of loss with respect to bias
		params: 
		lr - float - learning rate
		gradient - numpy array - gradient of loss wrt to output of previous layer
		'''
	
		#d_loss_d_outfc is the derivative of loss wrt to output of fully connected layer 
		#batch_size x number of classes
		d_loss_d_outfc = gradient

		#derivative of output of fully connected layer wrt to weights/bias/input
		d_outfc_d_weight = self.prev_input.T #(nodes x batch_size)
		d_outfc_d_bias = 1
		d_outfc_d_x = self.weights.T #(output_dim x nodes)

		#derivative of loss wrt to weights/bias/input
		#d_loss/d_weight = d_loss/d_outfc * d_outfc/d_weight
		#d_loss/d_bias = d_loss/d_outfc * d_outfc/d_bias
		#d_loss/d_x = d_loss/d_outfc * d_outfc/d_x
		d_loss_d_weights = np.matmul(d_outfc_d_weight, d_loss_d_outfc)
		d_loss_d_bias = np.sum(np.multiply(d_loss_d_outfc, d_outfc_d_bias), axis=0)

		#batch size x flattened nodes
		d_loss_d_x = np.matmul(d_loss_d_outfc, d_outfc_d_x)#.reshape(self.prev_shape)
					

		#update weights/bias via gradient descent
		self.weights -= np.multiply(lr,d_loss_d_weights)
		if(self.is_bias):
			self.bias -= np.multiply(lr, d_loss_d_bias)
		return d_loss_d_x, d_loss_d_weights, d_loss_d_bias


