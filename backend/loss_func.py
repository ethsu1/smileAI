import numpy as np
from layers import *
def cross_entropy(output, labels):
	'''
	calculates the cross entropy loss (negative log likelihood and softmax)
	return:
	loss - float - total loss on the batch of data
	correct - int - number of correctly predicted data points
	softmax_output - numpy array -  logsoftmax output
	params: 
	output - numpy array - output from fully connected layer (batch size x num classes)
	labels - numpy array - array of ground truth labels
	'''

	layer = layers()
	softmax_output = layer.logsoftmax(output)
	loss = 0
	correct = 0
	for i,label in enumerate(labels):
		predicted_label = np.argmax(softmax_output[i])
		if(predicted_label == label):
			correct += 1
		loss += -softmax_output[i][label]
	loss /= labels.shape[0]
	return loss, correct, softmax_output

