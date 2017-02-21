import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	y = sigmoid(x)
	return y * (1.0 - y)

class ToyNeuralNetwork(object):
	def __init__(self, inputs, hidden_one, hidden_two, output):
		self.inputs = inputs + 1
		self.hidden_one = hidden_one
		self.hidden_two = hidden_two
		self.output = output

		# set up array of 1s for activations
		self.nni = [1.0] * self.inputs
		self.nnh_one = [1.0] * self.hidden_one
		self.nnh_two = [1.0] * self.hidden_two
		self.nno = [1.0] * self.output

		# create randomized weights
		self.nnwi = np.random.randn(self.inputs, self.hidden_one)
		self.nnwh = np.random.randn(self.hidden_one, self.hidden_two)
		self.nnwo = np.random.randn(self.hidden_two, self.output)

		# create arrays of 0 for changes
		self.nnci = np.zeros((self.inputs, self.hidden_one))
		self.nnch = np.zeros((self.hidden_one, self.hidden_two))
		self.nnco = np.zeros((self.hidden_two, self.output))

	def feedForward(self, inputs):
		if len(inputs) != self.inputs - 1:
			raise ValueError('Wrong number of inputs!')

		# input activations
		for i in range(self.inputs - 1):
			self.nni[i] = inputs[i]

		# hidden_one activations
		for j in range(self.hidden_one):
			feedback = 0.0
			for i in range(self.inputs):
				feedback += self.nni[i] * self.nnwi[i][j]
			self.nnh_one[j] = sigmoid(feedback)

		# hidden_two activations
		for q in range(self.hidden_two):
			feedback = 0.0
			for p in range(self.hidden_one):
				feedback += self.nnh_one[p] * self.nnwh[p][q]
			self.nnh_two[q] = sigmoid(feedback)

		# output activations
		for k in range(self.output):
			feedback = 0.0
			for j in range(self.hidden_two):
				feedback += self.nnh_two[j] * self.nnwo[j][k]
			self.nno[k] = sigmoid(feedback)
		return self.nno[:]

	def backPropagate(self, targets, N):
		if len(targets) != self.output:
			raise ValueError('Wrong number of targets')

		# calculate error terms for output
		# delta tells you whihch direction to change the weights
		output_deltas = [0.0] * self.output
		for k in range(self.output):
			error = -(targets[k] - self.nno[k])
			output_deltas[k] = dsigmoid(self.nno[k]) * error

		# calculate error terms for hidden_two
		# delta tells you which direction to change the weights
		hidden_two_deltas = [0.0] * self.hidden_two
		for j in range(self.hidden_two):
			error = 0.0
			for k in range(self.output):
				error += output_deltas[k] * self.nnwo[j][k]
			hidden_two_deltas[j] = dsigmoid(self.nnh_two[j]) * error

		# calculate error terms for hidden_one
		# delta tells you which direction to change the weights
		hidden_one_deltas = [0.0] * self.hidden_one
		for j in range(self.hidden_one):
			error = 0.0
			for k in range(self.hidden_two):
				error += hidden_two_deltas[k] * self.nnwh[j][k]
			hidden_one_deltas[j] = dsigmoid(self.nnh_one[j]) * error

		# update the weights connecting hidden_two to output
		for j in range(self.hidden_two):
			for k in range(self.output):
				change = output_deltas[k] * self.nnh_two[j]
				self.nnwo[j][k] -= N * change + self.nnco[j][k]
				self.nnco[j][k] = change

		# update the weights connecting hidden_one to hidden_two
		for j in range(self.hidden_one):
			for k in range(self.hidden_two):
				change = hidden_two_deltas[k] * self.nnh_one[j]
				self.nnwh[j][k] -= N * change + self.nnch[j][k]
				self.nnch[j][k] = change

		# update the weights connecting input to hidden_one
		for i in range(self.inputs):
			for j in range(self.hidden_one):
				change = hidden_one_deltas[j] * self.nni[i]
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
