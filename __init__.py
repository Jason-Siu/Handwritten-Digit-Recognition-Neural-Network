from sklearn import datasets # import dataset
import numpy as np
from matplotlib import pyplot as plt # for potentially graphing the plot of the error

# seed for randomness
np.random.seed(15)
training_data_size = 1500
# only prints to 10 decimal places
np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

# initializes weights. This could've been done better
# by saying weights = np.random.randn(weightDimension1, weightDimension2)
def initialize_weights(x,y):
    weight = []
    for i in range(x):
        inner = []
        for j in range(y):
            inner.append(np.random.randn())
        weight.append(inner)
    return weight

# again this could been done better by saying
# bias = np.random.randn(biasDimension1, biasDimension2)
def getBiases(size):
    array = []
    bias = np.random.randn()
    for i in range(size):
        array.append(bias)
    return array

# sigmoid function: gives curviness and flexibility to our classifier
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# gets inputs returns as 2d array
def getTestData():
    datal = []
    for i in range(training_data_size):
        datal.append(data[i])
    return datal

# generates to targets for the test. This could've been done more elegeantly
# my lack of python knowledge back then is showing by examples like these
def getTestTarget():
    datal = []
    data2 = []
    for i in range(training_data_size):
        datal.append(target[i])
    for i in range(training_data_size):
        if target[i] == 0:
            data2 += [[1,0,0,0,0,0,0,0,0,0]]
        elif target[i] == 1:
            data2 += [[0,1,0,0,0,0,0,0,0,0]]
        elif target[i] == 2:
            data2 += [[0,0,1,0,0,0,0,0,0,0]]
        elif target[i] == 3:
            data2 += [[0,0,0,1,0,0,0,0,0,0]]
        elif target[i] == 4:
            data2 += [[0,0,0,0,1,0,0,0,0,0]]
        elif target[i] == 5:
            data2 += [[0,0,0,0,0,1,0,0,0,0]]
        elif target[i] == 6:
            data2 += [[0,0,0,0,0,0,1,0,0,0]]
        elif target[i] == 7:
            data2 += [[0,0,0,0,0,0,0,1,0,0]]
        elif target[i] == 8:
            data2 += [[0,0,0,0,0,0,0,0,1,0]]
        elif target[i] == 9:
            data2 += [[0,0,0,0,0,0,0,0,0,1]]
    return data2

# maximum number is what the computer guesses
def computerGuess(x):
    max_index = 0
    for i in range(9):
        if x[max_index] < x[i + 1]:
            max_index = i + 1
    return max_index
        
# load dataset
digits = datasets.load_digits()
data = digits.data
target = digits.target

test_data = np.array(getTestData())
test_target = np.array(getTestTarget())


# learning rate
alpha = .005

# weights for the 2 hidden layers and the output layer
syn0 = np.array(initialize_weights(64, 16))
syn1 = np.array(initialize_weights(16, 16))
syn2 = np.array(initialize_weights(16,10))

# biases for the layers
first_bias = np.array(getBiases(16))
second_bias = np.array(getBiases(16))
third_bias = np.array(getBiases(10))

#training loop
def train(x, syn0, syn1, syn2, first_bias, second_bias, third_bias):
    l0 = test_data
    
    #forward pass
    l1 = nonlin(np.dot(l0, syn0) + first_bias)
    l2 = nonlin(np.dot(l1, syn1) + second_bias)
    l3 = nonlin(np.dot(l2, syn2) + third_bias)
    
    # backprop
    l3_error = test_target - l3
    if x % 10000 == 0:
        print(np.mean(l3_error))
    l3_delta = l3_error * nonlin(l3, deriv = True)
    
    # third_bias += alpha * l3_delta
    
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * nonlin(l2,deriv=True)
    
    # second_bias += alpha * l2_delta
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv = True)
    # first_bias += alpha * l1_delta
    
    syn2 += alpha * l2.T.dot(l3_delta)
    syn1 += alpha * l1.T.dot(l2_delta)
    syn0 += alpha * l0.T.dot(l1_delta)
    
# training loop
for i in range(100000):
    train(i, syn0, syn1, syn2, first_bias, second_bias, third_bias)

# testing model
success = 0
for i in range(1796):
    l0 = data[i]
    l1 = nonlin(np.dot(l0,syn0) + first_bias)
    l2 = nonlin(np.dot(l1,syn1) + second_bias)
    l3 = nonlin(np.dot(l2,syn2) + third_bias)
    
    if(str(computerGuess(l3)) == str(target[i])):
        success += 1
    else:
        print(i)

# prints the proportion of correct guesses and total
# this got over 95% accuracy, however, it was probably most likely due to overfitting.
# this neural network doesn't include dropout, or relu activation
# this is a very naive approach for neural networks, but regardless it works
print(success / 1796)
