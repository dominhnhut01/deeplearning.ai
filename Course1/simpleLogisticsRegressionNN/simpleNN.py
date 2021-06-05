import numpy as np
import h5py
from lr_utils import load_dataset
import matplotlib.pyplot as plt

def flattenDataset(data):
	height = data.shape[1]
	width = data.shape[2]
	m_train = data.shape[0]
	newData = np.reshape(data,(m_train,-1), order="C").T
	newData = newData/255
	return newData

def initParam(X_train):
	w = np.zeros((X_train.shape[0], 1))	#X_train.shape[0] is the number of features
	b =0.0

	return w, b

def sigmoid(mat):
	return 1/(1+np.exp(-mat))

def propagation(X_train,Y_train, w, b):
	m = Y_train.shape[1]

	#Forward propagation
	A = sigmoid(np.dot(w.T, X_train) + b)
	lost = -np.dot(Y_train, np.log(A).T) - np.dot((1-Y_train), np.log(1-A).T)
	cost = 1/m * np.sum(lost)

	#Backward propagation
	dw = 1/m * np.dot(X_train, (A-Y_train).T)
	db = 1/m * np.sum(A-Y_train)

	grad = {
		"dw": dw,
		"db": db
	}

	return cost, grad

def optimize(X,Y,w,b,num_iterations = 1000, learning_rate = 0.009, print_step = False):
	costs = []
	for i in range(num_iterations):
		cost, grad = propagation(X,Y,w,b)

		dw = grad["dw"]
		db = grad["db"]

		w = w - learning_rate*dw
		b = b - learning_rate*db
		if i %10 == 0:
			costs.append(cost) 
			if print_step:
				print("The cost of {}th step is: {}".format(i, cost))

	return w,b, costs

def predict(w, b, X):
	A = sigmoid(np.dot(w.T, X) + b)

	m_train = X.shape[1]
	Y_prediction = np.zeros((1,m_train))

	for i in range(m_train):
		if A[0,i] > 0.5:
			Y_prediction[0,i] = 1
		else:
			Y_prediction[0,i] = 0

	return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.009, print_step = False):
	X_train = flattenDataset(X_train)
	X_test = flattenDataset(X_test)

	w, b = initParam(X_train)
	w, b, costs = optimize(X_train, Y_train, w, b, num_iterations, learning_rate, print_step)

	Y_prediction_train = predict(w, b, X_train)
	Y_prediction_test = predict(w, b, X_test)

	result = {
		"costs": costs,
		"w": w, 
		"b": b,
		"Y_prediction_test": Y_prediction_test,
		"Y_prediction_train": Y_prediction_train,
		"learning_rate": learning_rate,
		"num_iterations": num_iterations
	}
	return result

if __name__ == '__main__':
	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, _ = load_dataset()
	result = model(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, num_iterations = 2000, learning_rate = 0.009, print_step = False)
	Y_prediction_train = result["Y_prediction_train"]
	Y_prediction_test = result["Y_prediction_test"]
	count = 0
	for i in range(Y_prediction_test.shape[1]):
		if Y_test_orig[0,i] == Y_prediction_test[0,i]:
			count+=1

	print("Accuracy is: {} {}".format(count/Y_prediction_test.shape[1], count))
