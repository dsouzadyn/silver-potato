from toy_neural import *
from data_prepper import *


# The neural net takes 3 params:
# 1. No. of inputs
# 2. No. of hidden inputs
# 3. No. of outputs
# in that order

test_nn_and = ToyNeuralNetwork(2,4,1)
test_nn_xor = ToyNeuralNetwork(2,3,1)
test_nn_or = ToyNeuralNetwork(2,2,1)

training_and = get_and_training()
training_or = get_or_training()
training_xor = get_xor_training()


test_data = [[0, 1], [1, 0], [0, 0], [1, 1]]


print "Training with AND Gate data:"
test_nn_and.train(training_and)
print "Predicting test data:"
print test_nn_and.predict(test_data)

print "Training with OR Gate data:"
test_nn_or.train(training_or)
print "Predicting test data:"
print test_nn_or.predict(test_data)

print "Training with XOR Gate data:"
test_nn_xor.train(training_xor)
print "Predicting test data:"
print test_nn_xor.predict(test_data)

print "Done!"
