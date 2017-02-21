import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	y = sigmoid(x)
	return y * (1.0 - y)

class ToyNeuralNetwork(object):
	def __init__(self, inputs, hidden, output):
		self.inputs = inputs + 1
		self.hidden = hidden
		self.output = output
		
		# set up array of 1s for activations
		self.nni = [1.0] * self.inputs
		self.nnh = [1.0] * self.hidden
		self.nno = [1.0] * self.output

		# create randomized weights
		self.nnwi = np.random.randn(self.inputs, self.hidden)
		self.nnwo = np.random.randn(self.hidden, self.output)

		# create arrays of 0 for changes
		self.nnci = np.zeros((self.inputs, self.hidden))
		self.nnco = np.zeros((self.hidden, self.output))

	def feedForward(self, inputs):
		if len(inputs) != self.inputs - 1:
			raise ValueError('Wrong number of inputs!')

		# input activations
		for i in range(self.inputs - 1):
			self.nni[i] = inputs[i]

		# hidden activations
		for j in range(self.hidden):
			feedback = 0.0
			for i in range(self.inputs):
				feedback += self.nni[i]* self.nnwi[i][j]
			self.nnh[j] = sigmoid(feedback)

		# output activations
		for k in range(self.output):
			feedback = 0.0
			for j in range(self.hidden):
				feedback += self.nnh[j] * self.nnwo[j][k]
			self.nno[k] = sigmoid(feedback)
		return self.nno[:]

	def backPropagate(self, targets, N):
		if len(targets) != self.output:
			raise ValueError('Wrong number of targets')

		# calculate error terms for output
		# the delta tells you whihch direction to change the weights
		output_deltas = [0.0] * self.output
		for k in range(self.output):
			error = -(targets[k] - self.nno[k])
			output_deltas[k] = dsigmoid(self.nno[k]) * error

		# calculate error terms for hidden
		# delta tells you which direction to change the weights
		hidden_deltas = [0.0] * self.hidden
		for j in range(self.hidden):
			error = 0.0
			for k in range(self.output):
				error += output_deltas[k] * self.nnwo[j][k]
			hidden_deltas[j] = dsigmoid(self.nnh[j]) * error

		# update the weights connecting hidden to output
		for j in range(self.hidden):
			for k in range(self.output):
				change = output_deltas[k] * self.nnh[j]
				self.nnwo[j][k] -= N * change + self.nnco[j][k]
				self.nnco[j][k] = change

		# update the weights connecting input to hidden
		for i in range(self.inputs):
			for j in range(self.hidden):
				change = hidden_deltas[j] * self.nni[i]
				self.nnwi[i][j] -= N * change + self.nnci[i][j]
				self.nnci[i][j] = change

		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error += 0.5 * (targets[k] - self.nno[k]) ** 2

		return error

	def train(self, patterns, iterations = 3000, N = 0.0002):
		# N: learning rate
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.feedForward(inputs)
				error = self.backPropagate(targets, N)
			if i % 500 == 0:
				print 'error %-.5f' % error

	def predict(self, X):
		predictions = []
		for p in X:
			predictions.append(int(np.round(self.feedForward(p))))
		return predictions

