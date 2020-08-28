import numpy as np
import math
class convolution:
	def __init__(self, num_filters,kernel_size,padding, stride=1,bias=True):
		'''
		create a convolutional layer
		weights and biases are initialized by sampling a uniform distribution U(-k,k) where k = 1/(depth*kernel_size*kernel_size)
		num_filters - int - number of filters
		kernel_size - int - size of convolutional kernel (window size)
		padding - int - amount of zero padding (added to both sides)
		stride - int - stride (default: 1)
		'''
		self.num_filters = num_filters
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.filters = None
		#one bias parameter per fitler
		self.is_bias = bias
		self.bias = None

	def forward(self, matrix):
		'''
		matrix - numpy array - img/matrix of 4D shape (batch_size,depth/channel, height, width)
		'''
		self.prev_matrix = matrix
		if(len(matrix.shape) != 4):
			raise ValueError("Your image dimensions need to be 4D (batch_size, depth/channel, height, width")
		if(self.filters is None):
			#(number of filters, depth/channel, kernel_size, kernel_size)
			k = 1/(matrix.shape[1] * self.kernel_size * self.kernel_size)
			self.filters = np.random.uniform(low=-math.sqrt(k), high=math.sqrt(k),size=(self.num_filters, matrix.shape[1],self.kernel_size, self.kernel_size))
		if(self.bias is None and self.is_bias):
			k = 1/(matrix.shape[1] * self.kernel_size * self.kernel_size)
			self.bias = np.random.uniform(low=-math.sqrt(k),high=math.sqrt(k),size=(self.num_filters,))
		elif(self.is_bias == False):
			self.bias = np.zeros((num_filters,))
		#zero pad image, round down 
		o_height = math.floor((matrix.shape[2] - self.kernel_size + 2*self.padding)/self.stride) + 1
		o_width = math.floor((matrix.shape[3] - self.kernel_size + 2*self.padding)/self.stride) + 1

		o_depth = self.num_filters
		#zero pad image
		padded_matrix = []
		for i,sample in enumerate(matrix):
			padded_matrix.append([])
			for depth in range(matrix.shape[1]):
				padded_matrix[i].append(np.pad(sample[depth,:,:],(self.padding,self.padding), 'constant', constant_values=(0,0)))
		padded_matrix = np.asarray(padded_matrix)
		self.prev_matrix_padded = padded_matrix
		#(batch_size,depth/channels,height, width)
		output = np.zeros((padded_matrix.shape[0], o_depth, o_height, o_width))
		
		depth = padded_matrix.shape[1]
		height = padded_matrix.shape[2] 
		width = padded_matrix.shape[3]
		#iterate over each image in the batch
		for i,sample in enumerate(padded_matrix):
			for kernel_num, kernel in enumerate(self.filters):
				#iterate over channels
				for z in range(depth):
					out_x = 0
					for x in range(0, height, self.stride):
						out_y = 0
						for y in range(0, width,self.stride):
							if(out_y < o_width and out_x < o_height):
								output[i][kernel_num][out_x][out_y] += (np.sum(np.multiply(kernel[z,:,:,], sample[z,x:x+self.kernel_size,y:y+self.kernel_size])))
							out_y += 1
						out_x += 1
				#add bias to output after computing convolution for each channel
				out_x = 0
				while(out_x < o_height):
					out_y = 0
					while(out_y < o_width):
						output[i][kernel_num][out_x][out_y] += self.bias[kernel_num]
						out_y += 1
					out_x += 1
		return output

	def backprop(self, lr, gradient):
		'''
		perform backpropagation to adjust weights and bias of convolutional layer
		return:
		d_loss_d_x - numpy array - gradient of loss with respect to convolutional layer inputs
		d_loss_d_filters - numpy array - gradient of loss with respect to convolutional filters
		d_loss_d_bias - numpy array - gradient of loss with respect to bias
		params: 
		lr - float - learning
		gradient - numpy array - gradient of loss wrt to output of previous layer
		'''
		#derivative of loss wrt to output of convolutional layer
		d_loss_d_out = gradient


		padded_d_loss_d_out = []
		for i,sample in enumerate(gradient):
			padded_d_loss_d_out.append([])
			for depth in range(gradient.shape[1]):
				padded_d_loss_d_out[i].append(np.pad(sample[depth,:,:],(self.padding,self.padding), 'constant', constant_values=(0,0)))
		padded_d_loss_d_out = np.asarray(padded_d_loss_d_out)

		#derivative of loss wrt to weights/bias/inputs
		#number of filters x depth/channel x kernel_size x kernel_size
		d_loss_d_filters = np.zeros(self.filters.shape)
		d_loss_d_bias = np.zeros(self.bias.shape)
		#d_loss_d_x were that very first convolution layer, it should be the shape of input data (batch_size x channels x width x height)
		d_loss_d_x = np.zeros(self.prev_matrix.shape)

		#derivative of output of conv layer wrt to filters/bias/inputs
		d_convout_d_filters = self.prev_matrix_padded
		d_convout_d_bias = 1
		d_convout_d_x = np.copy(self.filters)
		d_convout_d_x = np.flip(d_convout_d_x, (2,3))
		

		grad_height = self.filters.shape[2]
		grad_width = self.filters.shape[3]
		
		#prev matrix is padded version of input matrix into conv layer
		#derivative of loss wrt to filters is convolution of forward pass input and gradient of loss from previous layer
		#iterate over each image in the batch
		kernel_size = d_loss_d_out.shape[2]
		for i in range(d_convout_d_filters.shape[0]):
			#iterate over number of filters
			for kernel_num in range(self.filters.shape[0]):
				#iterate over channels
				for z in range(self.filters.shape[1]):
					for x in range(0, grad_height, self.stride):
						for y in range(0, grad_width, self.stride):
							#derivative of loss wrt to weights 
							#d_loss/d_filters = d_loss/d_conv_out * d_conv_out/d_filters (which when rewritten is convolution of forward pass input and gradient of loss from previous layer)
							d_loss_d_filters[kernel_num][z][x][y] += (np.sum(np.multiply(d_convout_d_filters[i,z,x:x+kernel_size,y:y+kernel_size], d_loss_d_out[i,kernel_num,:,:])))
				
				#derivative of loss wrt to bias
				#d_loss/d_bias = d_loss/d_conv_out * d_convout/d_bias
				d_loss_d_bias[kernel_num] += np.sum(np.multiply(d_loss_d_out[i,kernel_num,:,:], d_convout_d_bias))

		#derivative of loss wrt to input is convolution of padded gradient of loss from previous layer and flipped filter
		o_height = d_loss_d_out.shape[2]
		o_width = d_loss_d_out.shape[3]
		depth = d_loss_d_x.shape[1]
		height = padded_d_loss_d_out.shape[2] 
		width = padded_d_loss_d_out.shape[3]
		#iterate over each image in the batch
		for i,sample in enumerate(padded_d_loss_d_out):
			for kernel_num, kernel in enumerate(d_convout_d_x):
				#iterate over channels
				for z in range(depth):
					out_x = 0
					for x in range(0, height, self.stride):
						out_y = 0
						for y in range(0, width,self.stride):
							if(out_y < o_width and out_x < o_height):
								#print(kernel[z,:,:,])
								d_loss_d_x[i][z][out_x][out_y] += (np.sum(np.multiply(kernel[z,:,:,], sample[kernel_num,x:x+self.kernel_size,y:y+self.kernel_size])))
							out_y += 1
						out_x += 1

		#derivative of loss wrt to inputs
		self.filters -= np.multiply(lr, d_loss_d_filters)
		if(self.is_bias):
			self.bias -= np.multiply(lr, d_loss_d_bias)
		return d_loss_d_x, d_loss_d_filters, d_loss_d_bias