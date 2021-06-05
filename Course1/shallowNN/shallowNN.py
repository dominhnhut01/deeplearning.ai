import numpy as np
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt

def initParam(X, neuron_num):
	m_train = X.shape[1]
	n_x = X.shape[0]

	W1 = np.random.randn(neuron_num, n_x)
	b1 = np.zeros((neuron_num, 1))
	W2 = np.random.randn(1,neuron_num)
	b2 = np.zeros((1,1))

	params = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}

	return params

def sigmoid(mat):
	return 1/(1+np.exp(-mat))

def forwardPropagation(X,params):
	W1 = params["W1"]
	b1 = params["b1"]
	W2 = params["W2"]
	b2 = params["b2"]

	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)

	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	cache = {
		"Z1": Z1,
		"A1": A1,
		"Z2": Z2,
		"A2": A2
	}

	return A2, cache

def computeCost(A2, Y):
	
	m = Y.shape[1]
	cost = -1/m * (np.dot(Y, np.log(A2).T) + np.dot((1-Y), np.log(1-A2).T))
	return cost

def backwardPropagation(X,Y,params, cache):
	W1 = params["W1"]
	b1 = params["b1"]
	W2 = params["W2"]
	b2 = params["b2"]	

	Z1 = cache["Z1"]
	A1 = cache["A1"]
	Z2 = cache["Z2"]
	A2 = cache["A2"]

	m = X.shape[1]

	dZ2 = A2 - Y
	dW2 = 1/m * np.dot(dZ2,A1.T)
	db2 = 1/m * np.sum(dZ2, axis = 1, keepdims=True)

	dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1,2))
	dW1 = 1/m * np.dot(dZ1, X.T)
	db1 = 1/m * np.sum(dZ1, axis=1, keepdims= True)

	grads = {
		"dW2": dW2,
		"db2": db2,
		"dW1": dW1,
		"db1": db1
	}

	return grads

def updateParams(X,Y,params, learning_rate = 1.2):
	W1 = params["W1"]
	b1 = params["b1"]
	W2 = params["W2"]
	b2 = params["b2"]

	A2, cache = forwardPropagation(X,params)
	cost = computeCost(A2,Y)
	grads = backwardPropagation(X,Y,params,cache)

	dW2 = grads["dW2"]
	db2 = grads["db2"]
	dW1 = grads["dW1"]
	db1 = grads["db1"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	params = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}

	return cost, params	

def model(X,Y, neuron_num = 4, iterations = 10000, learning_rate = 1.2):
	params = initParam(X, neuron_num)
	cost_list = []
	for i in range(iterations):
		cost, params = updateParams(X,Y,params, learning_rate)
		cost_list.append(cost)
	return cost_list, params

def predict(X, params):
	prediction, _ = forwardPropagation(X,params)
	prediction = (prediction > 0.5)
	return prediction

if __name__ == '__main__':
	X, Y = load_planar_dataset()
	#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
	#plt.show()
	cost_list, params = model(X, Y, neuron_num = 5)
	
	# Plot the decision boundary
	plot_decision_boundary(lambda x: predict(x.T, params), X, Y)
	plt.title("Decision Boundary for hidden layer size " + str(4))
	plt.show()
