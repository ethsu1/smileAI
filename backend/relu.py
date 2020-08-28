import numpy as np
class relu:
	def forward(self,matrix):
		'''
		apply relu activation function to matrix
		params:
		matrix - numpy array
		'''
		self.prev_input = matrix
		output = np.maximum(matrix, 0)
		return output

	def backprop(self, gradient):
		'''
		apply relu derivative to gradient of previous layer
		return:
		d_loss_d_x- numpy array - gradient of loss wrt to input into relu layer
		params:
		gradient - numpy array - gradient of loss wrt output of previous layer
		d_out_d_x - numpy array - gradient of output of previous layer wrt to inputs to the previous layer
		'''
		d_loss_d_x = np.copy(gradient)
		#d_loss/d_x = d_loss/d_out * d(max(0,x))/d_x
		d_loss_d_x[self.prev_input < 0] = 0
		return d_loss_d_x