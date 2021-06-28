from main_method_NN import *
from additional_method import *
import numpy as np

if __name__ == '__main__':
	np.random.seed(1)

	train_x, train_y, test_x, test_y = load_2D_dataset()
	print(train_x.shape)
	print(train_y.shape)
	m_train = train_x.shape[0]
	num_px = train_x.shape[1]
	m_test = test_x.shape[0]

	# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
	# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	# Standardize data to have feature values between 0 and 1.
	# train_x = train_x_flatten/255.0
	# test_x = test_x_flatten/255.0

	# layers_dims = (12288,20,7,5,1)
	layers_dims = [train_x.shape[0], 20, 3, 1]
	drop_out_keep_prob = [0.8,0.7,0.6,1]
	regularization_param = 0.01

	iteration = 20000
	params, costs_train, costs_test = model(train_x, train_y, test_x, test_y, layers_dims = layers_dims, drop_out_keep_prob = drop_out_keep_prob, iteration = iteration, learning_rate = 0.0075, regularization_param = regularization_param)
	plt.plot([iteration for iteration in range(0, iteration, 100)],costs_train)
	plt.plot([iteration for iteration in range(0, iteration, 100)],costs_test)

	train_prediction = predict(train_x, params, layers_dims)
	test_prediction = predict(test_x, params, layers_dims)

	train_accuracy = computeAccuracy(train_prediction, train_y)
	test_accuracy = computeAccuracy(test_prediction, test_y)

	print("Accuracy on training dataset is: {}".format(train_accuracy))
	print("Accuracy on testing dataset is: {}".format(test_accuracy))

	plt.title("Model without regularization")

	axes = plt.gca()
	axes.set_xlim([-0.75,0.40])
	axes.set_ylim([-0.75,0.65])
	plot_decision_boundary(test_x, test_y, params, layers_dims)

	plt.show()
