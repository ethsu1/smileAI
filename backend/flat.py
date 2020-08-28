import numpy as np
class flat:
	def forward(self, matrix):
		'''
		flattened matrix before fully connected layer
		params:
		matrix - numpy array
		'''
		self.prev_shape = matrix.shape
		flat = matrix.reshape((matrix.shape[0],-1))
		return flat
		#return np.asarray(flat)

	def backprop(self, gradient):
		'''
		reshape gradient to be shape of input into flat layer
		'''
		return gradient.reshape(self.prev_shape)