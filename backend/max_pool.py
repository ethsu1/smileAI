import numpy as np
import math
class max_pool:
	def __init__(self, size, padding=0, stride=None):
		'''
		create a max pooling layer
		size - int - window/kernel size
		padding - int - amount of zero padding, default: 0
		stride - int - stride, default: size
		'''
		if(padding > size/2):
			raise RuntimeError("Padding should be smaller than half the kernel size but got: kernel=" + str(size) + " padding="+str(padding))
		self.size = size
		self.padding = padding
		if(stride is None):
			self.stride = size
		else:
			self.stride = stride

	def forward(self, matrix):
		'''
		Apply 2D maxpooling on input matrix
		matrix - numpy array - input matrix to apply pooling to (4D: (batch_size, depth/channel, height, width))
		'''
		if(len(matrix.shape) != 4):
			raise ValueError("Your image dimensions need to be 4D (batch_size, depth/channel, height, width")
		#round down 
		o_height = math.floor((matrix.shape[2] - self.size + 2*self.padding)/self.stride) + 1
		o_width = math.floor((matrix.shape[3] - self.size + 2*self.padding)/self.stride) + 1

		o_depth = matrix.shape[1]

		if(self.padding > 0):
			#zero pad image
			padded_matrix = []
			for i,sample in enumerate(matrix):
				padded_matrix.append([])
				for depth in range(matrix.shape[1]):
					padded_matrix[i].append(np.pad(sample[depth,:,:],(self.padding,self.padding), 'constant', constant_values=(0,0)))
			padded_matrix = np.asarray(padded_matrix)
			matrix = padded_matrix
		self.prev_matrix = matrix
		output = np.zeros((matrix.shape[0], o_depth, o_height, o_width))
		#iterate over samples
		for i,sample in enumerate(matrix):
			for z in range(matrix.shape[1]):
				out_x = 0
				for x in range(0, matrix.shape[2], self.stride):
					out_y = 0
					for y in range(0, matrix.shape[3], self.stride):
						if(out_y < o_width and out_x < o_height):
							output[i][z][out_x][out_y] = np.amax(matrix[i,z,x:x+self.size,y:y+self.size])
						out_y += 1
					out_x+= 1
		return output


	def backprop(self, gradient):
		'''
		perform backpropagation to on max pooling layer
		return gradient to pass to other layers
		params: lr - float - learning rate
		gradient - numpy array - gradient of loss wrt to output of previous layer
		'''
		#derivative of loss wrt to output of max pooling layer
		d_loss_d_out = gradient

		gradient_height = gradient.shape[2]
		gradient_width = gradient.shape[3]
		#derivative of loss wrt to padded inputs to max pooling layer
		d_loss_d_xpad = np.zeros(self.prev_matrix.shape)
		for i, sample in enumerate(self.prev_matrix):
			for z in range(self.prev_matrix.shape[1]):
				grad_x = 0
				for x in range(0, self.prev_matrix.shape[2], self.stride):
					grad_y = 0
					for y in range(0, self.prev_matrix.shape[3], self.stride):
						if(grad_y < gradient_width and grad_x < gradient_height):
							idx = np.argmax(self.prev_matrix[i,z,x:x+self.size,y:y+self.size])
							idxs = np.unravel_index(idx, self.prev_matrix[i,z,x:x+self.size,y:y+self.size].shape)
							max_x,max_y = idxs
							d_loss_d_xpad[i][z][x+max_x][y+max_y] += d_loss_d_out[i][z][grad_x][grad_y]
						grad_y += 1
					grad_x += 1
		d_loss_d_x = d_loss_d_xpad
		#upad the padded gradient
		if(self.padding > 0):
			unpadded_matrix = []
			for i,sample in enumerate(d_loss_d_xpad):
				unpadded_matrix.append([])
				for depth in range(d_loss_d_xpad.shape[1]):
					unpadded_matrix[i].append(sample[depth,self.padding:-self.padding, self.padding:-self.padding])
			unpadded_matrix = np.asarray(unpadded_matrix)
			d_loss_d_x = unpadded_matrix
		return d_loss_d_x