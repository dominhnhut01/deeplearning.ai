import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(mat):
	return 1/(1+np.exp(-mat))

def relu(mat):
	return np.maximum(0.0, mat)

def reluBackward(dA, Z):
	Z[Z<=0] = 0
	Z[Z>0] = 1
	dZ = dA * Z
	return dZ
def sigmoidBackward(dA, A):
	return dA * A * (1-A)

def initializeParams(layers_dims):
	"""
	Initialize the weights and bias for the neural network

	Argumemts:
	layers_dims -- list containig dimensions of the layers

	Returns:
	params -- dictionary containing value of weight and bias for all layers
	"""

	L = len(layers_dims)
	params = {}
	for l in range(1, L):
		params["w" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
		params["b" + str(l)] = np.zeros((layers_dims[l], 1))

	return params

def linearForward(A_prev, w, b):
	"""
	Compute the Z at current layer

	Arguments:
	A_prev -- A value from previous layer
	w -- weight at current layer
	b --  bias at current layer

	Returns:
	Z -- Z value of current layer
	linearCache -- tuple containing (A,w,b)
	"""

	Z = np.dot(w, A_prev) + b

	return Z

def linearActivationForward(Z, activation):
	"""
	Compute the A at current layer

	Arguments:
	Z -- Z value at current layer
	activation -- a String declaring the name of desired activation function

	Returns:
	A -- A value at current layer
	"""

	if activation == "sigmoid":
		A = sigmoid(Z)
	elif activation == "relu":
		A = relu(Z)
	return A
def modelForwardPropagation(X, params, layers_dims):
	"""
	Operating the forward propagation step through L layer in one iteration

	Arguments:
	X -- data, numpy array (height * width * 3, m_training_examples)
	Y -- true label of the X data, numpy array (1, m_training_examples)
	params -- the dictionary storing the value of weights and biases
	layer_dims -- the dimension of the layers

	Returns:
	AL -- the A value at the Lth layer
	caches -- the list of all the (A_prev, Z, A, w,b) value at each layer
	"""
	L = len(layers_dims)
	caches = []
	A_prev = X
	for l in range(1, L-1):
		current_w = params["w" + str(l)]
		current_b = params["b" + str(l)]

		Z = linearForward(A_prev, current_w, current_b)
		A = linearActivationForward(Z, activation = "relu")
		cache = (A_prev, Z, A, current_w, current_b)
		A_prev = A
		caches.append(cache)

	current_w = params["w" + str(L-1)]
	current_b = params["b" + str(L-1)]

	Z = linearForward(A_prev, current_w, current_b)
	AL= linearActivationForward(Z, activation = "sigmoid")
	cache = (A_prev, Z, AL, current_w, current_b)
	caches.append(cache)
	return AL, caches

def computeCost (AL, Y):
	"""
	Compute the cost function

	Arguments:
	AL -- the A value at the Lth layer
	Y -- true label of the X data, numpy array (1, m_training_examples)

	Returns:
	cost -- the value of cost function
	"""
	m = Y.shape[1]

	cost = -1/m * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1 - AL).T))
	cost = np.squeeze(cost)

	return cost

def linearActivationBackward(dA, cache ,activation):
	"""
	Compute the derivative of cost with respect to Z at the current layer (only for layer 1 to layer L-1)

	Arguments:
	dA -- derivative of cost with respect to A at the current layer
	cache -- a tuple (A_prev, Z, A, w,b) value at the current layer
	activation -- a String declaring the name of the current activation function

	Returns:
	dA_prev -- derivative of cost with respect to A at the previous layer
	"""
	(A_prev, Z, A, w, b) = cache
	if activation == "sigmoid":
		dZ = sigmoidBackward(dA, A)
	if activation == "relu":
		dZ = reluBackward(dA, Z)
	return dZ

def linearBackward(dZ, cache):
	"""
	Compute the derivative of cost with respect to A at the previous layer, derivative of weights and biases at the current layer

	Arguments:
	dZ -- derivative of cost with respect to Z at current layer
	cache -- a tuple (A_prev, Z, A, w,b) value at the current layer

	Returns:
	dA_prev -- derivative of cost with respect to A at the previous layer
	dw -- derivative of the cost with respect to weight at the current layer
	db -- derivative of the cost with respect to bias at the current layer
	"""
	(A_prev, Z, A, w, b) = cache
	m = Z.shape[1]

	dA_prev = np.dot(w.T, dZ)
	dw = 1/m * np.dot(dZ, A_prev.T)
	db = 1/m * np.sum(dZ, axis = 1, keepdims = True)

	return dA_prev, dw, db


def modelBackwardPropagation(Y, AL, caches, layers_dims):
	"""
	Operating the back propagation step through L layers of each iteration

	Arguments:
	Y -- true label of the X data, numpy array (1, m_training_examples)
	AL -- the A value at the Lth layer at the previous forward propagation step
	caches -- the list of all the (A_prev, Z, A, w,b) value at each layer
	layers_dims -- the dimension of the layers

	Returns:
	grads -- the dictionary storing gradients dw, db, dA of each layer
	"""
	L = len(layers_dims)
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
	dZL = linearActivationBackward(dAL, caches[L-2], activation = "sigmoid")
	dA_prev, dw, db = linearBackward(dZL, caches[L-2])
	grads = {}
	grads["dA" + str(L-1)] = dAL
	grads["dw" + str(L-1)] = dw
	grads["db" + str(L-1)] = db

	for l in range(L-2, 0, -1):
		dA = dA_prev
		dZ = linearActivationBackward(dA, caches[l-1], activation = "relu")
		dA_prev, dw, db = linearBackward(dZ, caches[l-1])
		grads["dA" + str(l)] = dA
		grads["dw" + str(l)] = dw
		grads["db" + str(l)] = db
	return grads
def updateParams(params, grads, learning_rate):
	"""
	Update parameters after each iteration

	Arguments:
	params -- the dictionary storing the value of weights and biases
	grads -- the dictionary storing gradients dw, db, dA of each layer
	learning_rate -- learning rate

	Returns:
	params -- updated params variables
	"""
	L = int(len(params) / 2)

	for l in range(1, L+1):
		params["w" + str(l)] = params["w" + str(l)] - grads["dw" + str(l)] * learning_rate
		params["b" + str(l)] = params["b" + str(l)] - grads["db" + str(l)] * learning_rate

	return params
def model(X,Y,layers_dims, iteration = 1000, learning_rate=0.0075):
	"""
	Train the model to have appropriate weight and bias for a dataset

	Argumemts:
	X -- data, numpy array (height * width * 3, m_training_examples)
	Y -- true label of the X data, numpy array (1, m_training_examples)
	layer_dims -- the dimension of the layers
	iteration -- iteration
	learning_rate -- learning rate

	Returns:
	params -- appropriate parameters to predict
	costs -- costs of each 100 iterations
	"""

	params = initializeParams(layers_dims)
	costs = []
	for i in range(iteration):
		AL, caches = modelForwardPropagation(X, params, layers_dims)
		caches_size = []
		for cache in caches:
			cache_size = [x.shape for x in cache]
			caches_size.append(cache_size)
		cost = computeCost(AL, Y)
		grads = modelBackwardPropagation(Y, AL, caches, layers_dims)

		if i % 100 == 0:
			costs.append(cost)
			print("Cost of {}th iteration: {}".format(i+1, cost))
		params = updateParams(params, grads, learning_rate)
	return params, costs

def predict(X, params):
	"""
	Use to predict the label with the input parameters

	Argumemts:
	X -- data, numpy array (height * width * 3, m_training_examples)
	params -- the dictionary storing the value of weights and biases

	Returns:
	prediction -- the predicted label for X
	"""

	prediction, cache = modelForwardPropagation(X, params, layers_dims)
	prediction = (prediction > 0.5)
	return prediction

def computeAccuracy(prediction, Y):
	"""
	Compute accuracy

	Argumemts:
	prediction -- the predicted label for X
	Y -- true label of the X data, numpy array (1, m_training_examples)

	Returns:
	accuracy -- accuracy
	"""

	prediction = np.squeeze(prediction)
	Y = np.squeeze(Y)

	count = 0

	for i in range(len(prediction)):
		if prediction[i] == Y[i]:
			count+=1
	accuracy = count*1.0/len(Y)

	return accuracy
if __name__ == '__main__':
	np.random.seed(1)

	train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

	m_train = train_x_orig.shape[0]
	num_px = train_x_orig.shape[1]
	m_test = test_x_orig.shape[0]

	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.0
	test_x = test_x_flatten/255.0

	layers_dims = (12288,20,7,5,1)
	iteration = 2500
	params, costs = model(train_x, train_y, layers_dims = layers_dims, iteration = iteration, learning_rate = 0.0075)
	plt.plot([iteration for iteration in range(0, iteration, 100)],costs)

	train_prediction = predict(train_x, params)
	test_prediction = predict(test_x, params)

	train_accuracy = computeAccuracy(train_prediction, train_y)
	test_accuracy = computeAccuracy(test_prediction, test_y)

	print("Accuracy on training dataset is: {}".format(train_accuracy))
	print("Accuracy on testing dataset is: {}".format(test_accuracy))
	plt.show()
