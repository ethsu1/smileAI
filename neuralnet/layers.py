import numpy as np
class layers:
	def __init__(self):
		return

	def logsoftmax(self, matrix):
		'''
		apply softmax function (subtract max and apply log for numerical stability)
		params:
		matrix - numpy array
		'''
		x = matrix - matrix.max(axis=-1, keepdims=True)
		y = np.exp(x)
		#log division rule to prevent overflow
		return x  - np.log(y.sum(axis=1, keepdims=True))
		#return np.log(y / y.sum(axis=-1, keepdims=True))

	def gradient_logsoftmax(self, logsoftmax_out, labels):
		'''
		calculuate gradient of logsoftmax wrt to input into logsoftmax
		params:
		logsoftmax_out - numpy array - output of logsoftmax
		labels - numpy array - one hot encoded ground truth labels
		'''
		return (np.exp(logsoftmax_out) - labels)/len(labels)